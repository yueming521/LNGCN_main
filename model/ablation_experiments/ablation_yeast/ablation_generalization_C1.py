import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import random
import importlib
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader, Dataset
import dgl
from dgl.nn import GraphConv, Set2Set

import gc
from tqdm import tqdm
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _SCRIPT_DIR.parent
_GRAND_PARENT_DIR = _PARENT_DIR.parent
for _path in (_SCRIPT_DIR, _PARENT_DIR, _GRAND_PARENT_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

_DATA_MODULE_CANDIDATES = [
    os.getenv("FANHUA_DATA_MODULE"),
    "ablation_generalization_data"
]

_ACTIVE_DATA_MODULE_NAME = None
for _candidate in _DATA_MODULE_CANDIDATES:
    if not _candidate:
        continue
    try:
        data_module = importlib.import_module(_candidate)
        _ACTIVE_DATA_MODULE_NAME = _candidate
        break
    except ImportError:
        continue

if _ACTIVE_DATA_MODULE_NAME is None:
    data_module = importlib.import_module("ablation_generalization_data")
    _ACTIVE_DATA_MODULE_NAME = "ablation_generalization_data"

print(f"📦 Using data module: {_ACTIVE_DATA_MODULE_NAME}")
try:
    sys.path.insert(0, 'LNGCN_main/results/ablation')
    from ablation_models_v3_dc import AblationGCN
    print("✅ Successfully imported AblationGCN model")
except ImportError as e:
    print(f"❌ Failed to import AblationGCN: {e}")
    print("Please ensure ablation_models_v3.py exists and contains the AblationGCN class")
    sys.exit(1)


from torch.utils.checkpoint import checkpoint

nhid = 256
nhidh = 128
nhidhh = 64
dropout = 0.4
time_steps = 5
ode_unfolds = 2
EVAL_RANDOM_SEED = int(os.getenv("FANHUA_EVAL_SEED", "42"))

def setup_eval_seed(seed: int):
    if hasattr(data_module, "setup_seed"):
        data_module.setup_seed(seed)
    else:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if os.getenv("FANHUA_ENFORCE_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"🎲 Generalization eval random seed: {seed}")

setup_eval_seed(EVAL_RANDOM_SEED)

DEFAULT_MODEL_PATHS = {
    'C1_AUPRC': 'LNGCN_main/results/ablation/ablation_human/C1_NoLTCDense/C1_best_auprc.pt',
    'C1_AUROC': 'LNGCN_main/results/ablation/ablation_human/C1_NoLTCDense/C1_best_auroc.pt',
}

C1_CONFIG = {
    'description': 'C1_NoLTCq: Remove only the LTC Dense layer, keep the LTC graph layers, replace with SimpleDense',
    'scientific_hypothesis': 'Validate the contribution of the LTC Dense layer while preserving temporal modeling via LTC graph layers',
    'config': {
        'use_cfc': True, 
        'cfc_type': 'original',
        'cfc_layers': 3,

        'use_ltc': True,
        'ltc_type': 'enhanced', 
        'ltc_layers': 2,
        'ltc_residual': True,
        'use_distance': True,

        'use_structure_enhance': True,
        'use_multi_scale': True,

        'pool_type': 'set2set',

        'fusion_type': 'symmetric',

        'use_ltc_dense': False,
        'fc_layers': 2,
        'dropout': dropout,
    }
}

FANHUA_OUTPUT_DIR = Path(os.getenv("FANHUA_OUTPUT_DIR", "LNGCN_main/results/ablation/ablation_yeast/yeast_ablation_generalization-C1")).resolve()
DEFAULT_RESULTS_BASENAME = os.getenv("FANHUA_RESULTS_BASENAME", "generalization_test_results.json")
DEFAULT_OUTPUT_FILE = FANHUA_OUTPUT_DIR / DEFAULT_RESULTS_BASENAME

DEFAULT_DATA_ROOT = Path(os.getenv(
    "FANHUA_DATA_ROOT","LNGCN_main/data/yeast"
))
DEFAULT_NEG_FILE = DEFAULT_DATA_ROOT / "yeast_neg.txt"
DEFAULT_POS_FILE = DEFAULT_DATA_ROOT / "yeast_pos.txt"
DEFAULT_DATA_FILES = [
    (str(DEFAULT_NEG_FILE), 0),
    (str(DEFAULT_POS_FILE), 1),
]

DEFAULT_DEVICE = os.getenv("FANHUA_DEVICE", "cuda:0")
DEFAULT_BATCH_SIZE = int(os.getenv("FANHUA_BATCH_SIZE", "256"))

def resolve_model_paths() -> dict:
    env_paths = os.getenv("FANHUA_MODEL_PATHS")
    if env_paths:
        try:
            mapping = json.loads(env_paths)
            resolved = {str(k): str(v) for k, v in mapping.items() if isinstance(k, str)}
            if resolved:
                print(f"📁 Using {len(resolved)} models provided by FANHUA_MODEL_PATHS")
                return resolved
        except Exception as exc:
            print(f"⚠️ Failed to parse FANHUA_MODEL_PATHS; falling back to default: {exc}")

    base_dir = os.getenv("FANHUA_MODEL_DIR")
    if base_dir:
        base_path = Path(base_dir)
        if base_path.is_dir():
            candidates = sorted(base_path.glob("*.pt"))
            if candidates:
                mapping = {candidate.stem: str(candidate.resolve()) for candidate in candidates}
                print(f"📁 Found {len(mapping)} model files in directory {base_path}")
                return mapping
            else:
                print(f"⚠️ FANHUA_MODEL_DIR={base_path} did not contain any .pt model files")
        else:
            print(f"⚠️ FANHUA_MODEL_DIR={base_path} is not a valid directory; falling back to default paths")

    print("ℹ️ Using built-in default model paths (set FANHUA_MODEL_PATHS or FANHUA_MODEL_DIR to customize)")
    return DEFAULT_MODEL_PATHS.copy()

class CfCCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, is_first_layer: bool = False):
        super().__init__()
        self.is_first_layer = is_first_layer
        input_dim = in_dim if is_first_layer else (in_dim + hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        self.f_head = nn.Linear(hidden_dim, hidden_dim)
        self.g_head = nn.Linear(hidden_dim, hidden_dim)
        self.h_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_prev, feat, t):
        if self.is_first_layer:
            B = self.backbone(feat)
        else:
            combined = torch.cat([x_prev, feat], dim=-1)
            B = self.backbone(combined)
        ft = torch.sigmoid(-self.f_head(B) * t)
        dynamic_feat = self.g_head(B)
        static_feat = self.h_head(B)
        return ft * dynamic_feat + (1 - ft) * static_feat

class PreprocessCfC(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cells = nn.ModuleList([
            CfCCell((in_dim - 1) if i == 0 else hidden_dim, hidden_dim, is_first_layer=(i == 0))
            for i in range(n_layers)
        ])

    def forward(self, feat):
        if feat.shape[1] > self.hidden_dim:
            base_feat = feat[:, :-1]
            timestep = feat[:, -1:].contiguous()
        else:
            base_feat = feat
            timestep = torch.zeros(feat.size(0), 1, device=feat.device, dtype=feat.dtype)

        h = torch.zeros(base_feat.size(0), self.hidden_dim, device=feat.device, dtype=feat.dtype)
        x_in = base_feat
        for i, cell in enumerate(self.cells):
            t = (i + 1) / float(self.n_layers)
            h = cell(h, x_in, t)
            x_in = h

        return torch.cat([h, timestep], dim=-1)

class offLTC(nn.Module):
    def __init__(self, nhid, ode_unfolds=ode_unfolds):
        super().__init__()
        self.nhid = nhid
        self.ode_unfolds = ode_unfolds
        self.tau = nn.Parameter(torch.empty(nhid))
        nn.init.uniform_(self.tau, 0.1, 10)

        self.w_gate = nn.Sequential(
            nn.Linear(nhid + 1, nhid),
            nn.LayerNorm(nhid),
            nn.Sigmoid()
        )
        self.w_hid = nn.Sequential(
            nn.Linear(nhid + 1, nhid),
            nn.LayerNorm(nhid),
            nn.Tanh()
        )
        self.transform = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.LayerNorm(nhid)
        )
        self.A = nn.Parameter(torch.ones(nhid))
        self.out = nn.Linear(nhid, nhid)

        nn.init.xavier_normal_(self.transform[0].weight)
        nn.init.xavier_normal_(self.w_gate[0].weight)
        nn.init.xavier_normal_(self.w_hid[0].weight)

    def off_compute_dynamics(self, x, t):
        t_broadcast = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype) * t
        x_t = torch.cat([x, t_broadcast], dim=1)
        gate = self.w_gate(x_t)
        hidden = self.w_hid(x_t)
        dynamic = self.transform(gate * hidden)
        return dynamic

    def off_ode_step(self, x, f, delta_t):
        tau = torch.relu(self.tau) + 1e-8
        tau_inv = 1.0 / tau
        numerator = x + delta_t * f * self.A
        denominator = 1 + delta_t * (tau_inv + f)
        return numerator / (denominator + 1e-12)

    def forward(self, t, x, dt=0.01):
        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            f = self.off_compute_dynamics(x, t)
            x = self.off_ode_step(x, f, delta_t)
            t += delta_t
        return self.out(x)

class LTCDense(nn.Module):
    def __init__(self, in_features, out_features, time_steps):
        super().__init__()
        self.ltc = offLTC(nhid=in_features, ode_unfolds=ode_unfolds)
        self.time_steps = time_steps
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        dt = 1.0 / self.time_steps
        for step in range(self.time_steps):
            t = step * dt
            x = self.ltc(t, x, dt)
        return self.fc(x)

class EnhancedDistanceLTC(nn.Module):
    def __init__(self, nhid, ode_unfolds=5):
        super().__init__()
        self.nhid = nhid
        self.ode_unfolds = ode_unfolds
        self.tau = nn.Parameter(torch.empty(nhid))
        nn.init.uniform_(self.tau, 0.5, 5)

        self.agg_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU()
        )
        self.fusion_layer = nn.Linear(nhid * 2, nhid)
        self.transform = nn.Linear(nhid, nhid)
        self.A = nn.Parameter(torch.ones(nhid))
        self.out = nn.Linear(nhid, nhid)

        nn.init.xavier_normal_(self.transform.weight)
        nn.init.xavier_normal_(self.agg_conv.weight)
        nn.init.xavier_normal_(self.fusion_layer.weight)

    def _extract_distance_features(self, features):
        distance_times = features[:, -1:]
        other_features = features[:, :-1]
        return other_features, distance_times

    def _compute_enhanced_gating(self, graph, x, distance_times):
        x_agg = self.agg_conv(graph, x)
        distance_encoded = self.distance_encoder(distance_times)
        combined = torch.cat([x_agg, distance_encoded], dim=1)
        fused = self.fusion_layer(combined)
        gate = torch.sigmoid(fused)
        return gate

    def _ode_step(self, x, f, delta_t):
        tau = torch.relu(self.tau) + 1e-8
        tau_inv = 1.0 / tau
        numerator = x + delta_t * f * self.A
        denominator = 1 + delta_t * (tau_inv + f)
        denominator = denominator + 1e-12
        return numerator / denominator

    def forward(self, graph, features, t=0.0, dt=0.01):
        if features.shape[1] == self.nhid + 1:
            x, distance_times = self._extract_distance_features(features)
        else:
            x = features if features.shape[1] == self.nhid else features[:, :self.nhid]
            distance_times = torch.zeros(features.size(0), 1, device=features.device, dtype=features.dtype)

        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            gate = self._compute_enhanced_gating(graph, x, distance_times)
            f = self.transform(gate)
            x = self._ode_step(x, f, delta_t)

        return self.out(x)

class EnhancedLTCLayer(nn.Module):
    def __init__(self, nhid, time_steps=time_steps, residual=True):
        super().__init__()
        self.enhanced_ltc = EnhancedDistanceLTC(nhid=nhid, ode_unfolds=ode_unfolds)
        self.graph_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        self.multi_scale_conv = nn.ModuleList([
            dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True),
            dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        ])
        self.scale_fusion = nn.Linear(nhid * 3, nhid)
        self.bn = nn.LayerNorm(nhid)
        self.relu = nn.ReLU()
        self.time_steps = time_steps
        self.residual = residual

        nn.init.xavier_normal_(self.graph_conv.weight)
        for conv in self.multi_scale_conv:
            nn.init.xavier_normal_(conv.weight)
        nn.init.xavier_normal_(self.scale_fusion.weight)

    def forward(self, graph, feat_with_t):
        if feat_with_t.shape[1] == self.enhanced_ltc.nhid + 1:
            input_feat = feat_with_t
        else:
            zeros_t = torch.zeros(feat_with_t.size(0), 1, device=feat_with_t.device, dtype=feat_with_t.dtype)
            input_feat = torch.cat([feat_with_t, zeros_t], dim=-1)

        ltc_feat = self.enhanced_ltc(graph, input_feat)
        residual = ltc_feat

        graph_feat = self.graph_conv(graph, ltc_feat)
        scale1 = self.multi_scale_conv[0](graph, ltc_feat)
        scale2 = self.multi_scale_conv[1](graph, scale1)
        fused = self.scale_fusion(torch.cat([ltc_feat, scale1, scale2], dim=-1))

        x = self.bn(graph_feat + fused)
        x = self.relu(x)

        if self.residual:
            x = self.relu(x + residual)

        return x


class GeneralizationDataset(Dataset):
    def __init__(self, data_files):
        self.samples = []
        self.load_data_from_files(data_files)

    def load_data_from_files(self, data_files):
        print(f"🔄 Loading generalization test data...")

        total_valid = 0
        total_invalid = 0

        exists_fn = getattr(data_module, "check_protein_exists_fast", None)
        if exists_fn is None:
            exists_fn = getattr(data_module, "check_protein_exists", None)
        if exists_fn is None:
            raise AttributeError("Data module lacks check_protein_exists, cannot validate protein IDs")

        for file_path, label in data_files:
            print(f"📁 Processing file: {file_path} (label: {label})")

            if not os.path.exists(file_path):
                print(f"⚠️ File does not exist, skipping: {file_path}")
                continue

            valid_count = 0
            invalid_count = 0

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 2:
                        p1, p2 = parts[0].strip(), parts[1].strip()

                        try:
                            if exists_fn(p1) and exists_fn(p2):
                                self.samples.append((p1, p2, label))
                                valid_count += 1
                            else:
                                invalid_count += 1
                                if hasattr(data_module, "append_fail_cache"):
                                    data_module.append_fail_cache(f"{p1}|{p2}", "protein-not-found", label="fanhua-dataset")
                        except Exception as e:
                            print(f"⚠️ Error checking proteins {p1}, {p2}: {e}")
                            invalid_count += 1
                            if hasattr(data_module, "append_fail_cache"):
                                data_module.append_fail_cache(f"{p1}|{p2}", f"protein-check-error: {e}", label="fanhua-dataset")
                    else:
                        print(f"⚠️ Invalid format on line {line_idx+1}, skipping: {line}")
                        invalid_count += 1

            print(f"   ✅ {file_path}: {valid_count} valid samples, {invalid_count} invalid samples")
            total_valid += valid_count
            total_invalid += invalid_count

        print(f"📊 Total: {total_valid} valid samples, {total_invalid} invalid samples")

        if len(self.samples) == 0:
            raise ValueError("No valid test samples found. Please check file format and protein IDs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_generalization_data(batch):
    result = data_module.collate_GCN(batch)
    if result is None:
        return None, None, None
    return result

def _ensure_dir(path):
    path_str = str(path)
    try:
        os.makedirs(path_str, exist_ok=True)
    except Exception:
        pass

def _clean_state_dict(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(k.startswith('module.') for k in keys):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict

def _load_model_weights(model: nn.Module, model_path: str, device: str) -> bool:
    try:
        ckpt = torch.load(model_path, map_location='cpu' if device.startswith('cpu') else device)
        if isinstance(ckpt, dict):
            cand_keys = ['state_dict', 'model_state', 'model', 'net', 'weights']
            sd = None
            for k in cand_keys:
                if k in ckpt and isinstance(ckpt[k], dict):
                    sd = ckpt[k]
                    break
            if sd is None:
                sd = ckpt
        else:
            sd = ckpt
        sd = _clean_state_dict(sd)
        model.load_state_dict(sd, strict=False)
        return True
    except Exception as e:
        print(f"❌ Failed to load model weights: {model_path} -> {e}")
        return False

def evaluate_model(model, test_loader, device, model_name="Model", save_txt=True):
    model.eval()
    all_predictions = []
    all_labels = []

    print(f"🧪 Evaluating {model_name}...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
            if batch[0] is None:
                continue

            batch_g1, batch_g2, batch_labels = batch
            batch_g1 = batch_g1.to(device)
            batch_g2 = batch_g2.to(device)
            batch_labels = batch_labels.to(device)

            fea1 = batch_g1.ndata['fea'].to(device, dtype=torch.float32)
            fea2 = batch_g2.ndata['fea'].to(device, dtype=torch.float32)

            outputs = model(batch_g1, batch_g2, fea1, fea2)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    if len(np.unique(all_labels)) < 2:
        print(f"⚠️ {model_name}: test set contains only one class; cannot compute full metrics")
        return None

    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_predictions)
    auprc = auc(recall_curve, precision_curve)
    auroc = roc_auc_score(all_labels, all_predictions)

    predictions_binary = np.array(all_predictions) > 0.5
    accuracy = accuracy_score(all_labels, predictions_binary)
    precision = precision_score(all_labels, predictions_binary, zero_division=0)
    recall = recall_score(all_labels, predictions_binary, zero_division=0)
    f1 = f1_score(all_labels, predictions_binary, zero_division=0)

    results = {
        'model_name': model_name,
        'auprc': float(auprc),
        'auroc': float(auroc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'num_samples': int(len(all_labels)),
        'num_positive': int(sum(all_labels)),
        'num_negative': int(len(all_labels) - sum(all_labels))
    }

    if save_txt:
        save_model_results_to_txt(results, model_name)

    print(f"📊 {model_name} evaluation results:")
    print(f"   AUPRC: {auprc:.4f}")
    print(f"   AUROC: {auroc:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")
    print(f"   Samples: {int(len(all_labels))} (positive: {int(sum(all_labels))}, negative: {int(len(all_labels) - sum(all_labels))})")

    return results

def save_model_results_to_txt(results, model_name):
    _ensure_dir(FANHUA_OUTPUT_DIR)
    filename = FANHUA_OUTPUT_DIR / f"generalization_test_{model_name.replace('/', '_').replace(':', '_')}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Protein interaction prediction model generalization test results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model name: {results['model_name']}\n")
        f.write(f"Test time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Metrics:\n")
        f.write(f"{'='*30}\n")
        f.write(f"AUPRC (area under PR curve):    {results['auprc']:.6f}\n")
        f.write(f"AUROC (area under ROC curve):   {results['auroc']:.6f}\n")
        f.write(f"Accuracy:                       {results['accuracy']:.6f}\n")
        f.write(f"Precision:                      {results['precision']:.6f}\n")
        f.write(f"Recall:                         {results['recall']:.6f}\n")
        f.write(f"F1 score:                       {results['f1']:.6f}\n")
        f.write(f"\n")
        f.write(f"Sample statistics:\n")
        f.write(f"{'='*30}\n")
        f.write(f"Total samples:   {results['num_samples']}\n")
        f.write(f"Positive samples:{results['num_positive']}\n")
        f.write(f"Negative samples:{results['num_negative']}\n")
        f.write(f"Positive ratio:  {results['num_positive']/results['num_samples']:.4f}\n")
        f.write(f"Negative ratio:  {results['num_negative']/results['num_samples']:.4f}\n")
        f.write(f"\n")
        f.write(f"Metric notes:\n")
        f.write(f"{'='*30}\n")
        f.write(f"AUPRC: Suitable for imbalanced datasets; higher is better\n")
        f.write(f"AUROC: Overall classification performance; higher is better\n")
        f.write(f"Accuracy: Proportion of correctly predicted samples\n")
        f.write(f"Precision: Proportion of predicted positives that are true positives\n")
        f.write(f"Recall: Proportion of true positives correctly predicted\n")
        f.write(f"F1 score: Harmonic mean of precision and recall\n")

    print(f"💾 {model_name} results saved to: {filename}")

def test_all_models(data_files, device=DEFAULT_DEVICE, batch_size=DEFAULT_BATCH_SIZE):
    print("🚀 Starting C1_NoLTCq generalization test...")
    print("🔄 Initializing data module...")
    data_module.init_graph_cache(max_size=50)
    dataset_meta = data_module.load_unified_dataset()
    ds_mode = dataset_meta.get('mode', 'unknown')
    if ds_mode == 'npz':
        ds_count = dataset_meta.get('count', '?')
        ds_root = dataset_meta.get('root', 'N/A')
    else:
        graphs = dataset_meta.get('graphs')
        ds_count = len(graphs) if graphs is not None else dataset_meta.get('count', '?')
        ds_root = dataset_meta.get('root', 'N/A')
    print(f"✅ Dataset loaded: mode={ds_mode}, graph count={ds_count}, root={ds_root}")

    test_dataset = GeneralizationDataset(data_files)

    try:
        if hasattr(data_module, 'batch_preload_coordinates'):
            all_ids = []
            for p1, p2, _y in test_dataset.samples:
                all_ids.append(p1)
                all_ids.append(p2)
            if all_ids:
                print("🧠 Preloading protein coordinates into cache (memory/disk)...")
                data_module.init_coord_cache()  # ensure cache is initialized
                data_module.batch_preload_coordinates(all_ids, max_workers=1, label="fanhua-preload", every=2000)
    except Exception as _e:
        print(f"⚠️ Coordinate preload failed (continuing): {_e}")

    if hasattr(data_module, 'make_collate_GCN'):
        collate_fn = data_module.make_collate_GCN(label="fanhua-eval", every=500)
    else:
        collate_fn = collate_generalization_data

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=False      
    )

    sample_batch = None
    sample_iter = iter(test_loader)
    while True:
        try:
            candidate_batch = next(sample_iter)
        except StopIteration:
            break
        if candidate_batch[0] is not None:
            sample_batch = candidate_batch
            break

    if sample_batch is None or sample_batch[0] is None:
        raise ValueError("Unable to obtain valid samples to infer input dimension")

    in_dim = sample_batch[0].ndata['fea'].shape[1]
    print(f"📏 Input feature dimension: {in_dim}")

    model_paths = resolve_model_paths()
    if not model_paths:
        print("⚠️ No model paths found for evaluation; check FANHUA_MODEL_PATHS / FANHUA_MODEL_DIR")
        return []

    all_results = []

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            continue

        print(f"\n🔄 Loading model: {model_name}")

        model = AblationGCN(in_dim=in_dim, nhid=nhid, dropout=dropout, time_steps=time_steps, experiment_config=C1_CONFIG)

        ok = _load_model_weights(model, model_path, device)
        if not ok:
            continue
        model = model.to(device)
        print(f"✅ Successfully loaded C1_NoLTCq model: {model_name}")

        results = evaluate_model(model, test_loader, device, model_name, save_txt=True)
        if results:
            all_results.append(results)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if hasattr(data_module, "clear_graph_cache"):
        data_module.clear_graph_cache()

    return all_results

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_results(results, output_file=None):
    output_path = Path(output_file) if output_file else DEFAULT_OUTPUT_FILE
    if not output_path.is_absolute():
        output_path = FANHUA_OUTPUT_DIR / output_path
    _ensure_dir(output_path.parent)

    print(f"\n💾 Saving results to {output_path}")

    results_converted = convert_numpy_types(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(results)
    summary_file = output_path.with_name(output_path.stem + '_summary.csv')
    df.to_csv(summary_file, index=False, encoding='utf-8')

    print("\n📊 C1_NoLTCq generalization test summary:")
    print("="*80)
    print(f"{'Model':<20} {'AUPRC':<8} {'AUROC':<8} {'Accuracy':<8} {'F1':<8} {'Samples':<8}")
    print("-"*80)

    auprc_scores = []
    auroc_scores = []

    for result in results:
        print(f"{result['model_name']:<20} {result['auprc']:<8.4f} {result['auroc']:<8.4f} "
              f"{result['accuracy']:<8.4f} {result['f1']:<8.4f} {result['num_samples']:<8}")
        auprc_scores.append(result['auprc'])
        auroc_scores.append(result['auroc'])

    print("-"*80)
    print(f"{'Mean':<20} {np.mean(auprc_scores):<8.4f} {np.mean(auroc_scores):<8.4f}")
    print(f"{'Std':<20} {np.std(auprc_scores):<8.4f} {np.std(auroc_scores):<8.4f}")
    print(f"{'Max':<20} {np.max(auprc_scores):<8.4f} {np.max(auroc_scores):<8.4f}")
    print(f"{'Min':<20} {np.min(auprc_scores):<8.4f} {np.min(auroc_scores):<8.4f}")

    print(f"\n✅ Detailed results saved to: {output_path}")
    print(f"✅ Summary table saved to: {summary_file}")

def main():
    data_files = list(DEFAULT_DATA_FILES)
    device = DEFAULT_DEVICE
    batch_size = DEFAULT_BATCH_SIZE
    output_file = DEFAULT_OUTPUT_FILE

    if device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️ CUDA not available, switching to CPU")
        device = 'cpu'

    print(f"🎯 Using device: {device}")
    neg_files = [p for p, lab in data_files if lab == 0]
    pos_files = [p for p, lab in data_files if lab == 1]
    print(f"📁 Negative files: {', '.join(neg_files) if neg_files else 'None'}")
    print(f"📁 Positive files: {', '.join(pos_files) if pos_files else 'None'}")
    print(f"📦 Batch size: {batch_size}")

    try:
        results = test_all_models(data_files, device, batch_size)
        if not results:
            print("❌ No model was successfully evaluated")
            return

        save_results(results, output_file)

        print("\n🎉 C1_NoLTCq generalization test completed!")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
