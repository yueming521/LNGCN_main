import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import random
import importlib
import pickle
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, roc_curve, matthews_corrcoef, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import dgl
from dgl.nn import GraphConv, Set2Set
import gc
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _SCRIPT_DIR.parent
for _path in (_SCRIPT_DIR, _PARENT_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

_DATA_MODULE_CANDIDATES = [
    os.getenv("FANHUA_DATA_MODULE"),
    "data_generalization"
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
    data_module = importlib.import_module("data_generalization")
    _ACTIVE_DATA_MODULE_NAME = "data_generalization"

print(f"📦 Using data module: {_ACTIVE_DATA_MODULE_NAME}")

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

    print(f"🎲 Generalization evaluation random seed: {seed}")


setup_eval_seed(EVAL_RANDOM_SEED)
DEFAULT_MODEL_PATHS = {
    'Fold1_AUPRC': 'LNGCN_main/results/balance_human/fold1_best_auprc.pt',
    'Fold1_AUROC': 'LNGCN_main/results/balance_human/fold1_best_auroc.pt',
    'Fold2_AUPRC': 'LNGCN_main/results/balance_human/fold2_best_auprc.pt',
    'Fold2_AUROC': 'LNGCN_main/results/balance_human/fold2_best_auroc.pt',
    'Fold3_AUPRC': 'LNGCN_main/results/balance_human/fold3_best_auprc.pt',
    'Fold3_AUROC': 'LNGCN_main/results/balance_human/fold3_best_auroc.pt',
    'Fold4_AUPRC': 'LNGCN_main/results/balance_human/fold4_best_auprc.pt',
    'Fold4_AUROC': 'LNGCN_main/results/balance_human/fold4_best_auroc.pt',
    'Fold5_AUPRC': 'LNGCN_main/results/balance_human/fold5_best_auprc.pt',
    'Fold5_AUROC': 'LNGCN_main/results/balance_human/fold5_best_auroc.pt'
}


FANHUA_OUTPUT_DIR = Path(os.getenv("FANHUA_OUTPUT_DIR", "LNGCN_main/results/generalization_yeast")).resolve()
DEFAULT_RESULTS_BASENAME = os.getenv("FANHUA_RESULTS_BASENAME", "generalization_test_results.json")
DEFAULT_OUTPUT_FILE = FANHUA_OUTPUT_DIR / DEFAULT_RESULTS_BASENAME

LAZY_CACHE_DIR = FANHUA_OUTPUT_DIR / "yeastlanjz"
PLOT_OUTPUT_DIR = FANHUA_OUTPUT_DIR / "huitu"
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
                print(f"📁 Using {len(resolved)} models provided by environment variable FANHUA_MODEL_PATHS")
                return resolved
        except Exception as exc:
            print(f"⚠️ Failed to parse FANHUA_MODEL_PATHS, falling back to default strategy: {exc}")

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
                print(f"⚠️ No .pt model files found in specified FANHUA_MODEL_DIR={base_path}")
        else:
            print(f"⚠️ FANHUA_MODEL_DIR={base_path} is not a valid directory, falling back to default paths")

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

class MyGCN(nn.Module):
    def __init__(self, in_dim, nhid=nhid, dropout=dropout, time_steps=time_steps):
        super(MyGCN, self).__init__()
        self.nhid = nhid
        self.cfc_preprocess = PreprocessCfC(in_dim=in_dim, hidden_dim=nhid, n_layers=3)
        self.ltc_conv1 = EnhancedLTCLayer(nhid=nhid, time_steps=time_steps, residual=True)
        self.ltc_conv2 = EnhancedLTCLayer(nhid=nhid, time_steps=time_steps, residual=True)
        self.structure_enhance = nn.Sequential(
            dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True),
            nn.LayerNorm(nhid),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.pool = dgl.nn.Set2Set(nhid, n_iters=3, n_layers=1)
        self.projection = nn.Linear(2 * nhid, nhid)
        self.fc1 = LTCDense(3 * nhid, nhidh, time_steps)
        self.fc2 = LTCDense(nhidh, nhidhh, time_steps)
        self.fc3 = nn.Linear(nhidhh, 2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.structure_enhance[0].weight)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, g1, g2, fea1, fea2):
        # Step 1: CfC preprocessing - keep residue-distance timestep features
        fea1 = checkpoint(self.cfc_preprocess, fea1, use_reentrant=False)
        fea2 = checkpoint(self.cfc_preprocess, fea2, use_reentrant=False)

        # Step 2: First enhanced LTC layer
        fea1 = self.ltc_conv1(g1, fea1)
        fea1 = self.dropout(fea1)
        fea2 = self.ltc_conv1(g2, fea2)
        fea2 = self.dropout(fea2)

        # Step 3: Second enhanced LTC layer
        fea1 = self.ltc_conv2(g1, fea1)
        fea2 = self.ltc_conv2(g2, fea2)

        # Step 4: Additional graph-structure enhancement
        enhanced_fea1 = self.structure_enhance[0](g1, fea1)
        enhanced_fea1 = self.structure_enhance[1](enhanced_fea1)
        enhanced_fea1 = self.structure_enhance[2](enhanced_fea1)
        enhanced_fea1 = self.structure_enhance[3](enhanced_fea1)

        enhanced_fea2 = self.structure_enhance[0](g2, fea2)
        enhanced_fea2 = self.structure_enhance[1](enhanced_fea2)
        enhanced_fea2 = self.structure_enhance[2](enhanced_fea2)
        enhanced_fea2 = self.structure_enhance[3](enhanced_fea2)

        # Residual connection
        fea1 = fea1 + enhanced_fea1
        fea2 = fea2 + enhanced_fea2
        
        fea1 = self.dropout(fea1)
        fea2 = self.dropout(fea2)

        # Step 5: Graph pooling
        g1.ndata['h'] = fea1
        g2.ndata['h'] = fea2
        hg1 = self.pool(g1, g1.ndata['h'])
        hg2 = self.pool(g2, g2.ndata['h'])
        
        # Clear graph node data to release memory
        del g1.ndata['h']
        del g2.ndata['h']

        # Step 6: Graph representation projection and fusion
        hg1 = self.projection(hg1)
        hg2 = self.projection(hg2)

        # === Optimization: Feature-level symmetric fusion ===
        
        # 1. Element-wise sum (Symmetric)
        hg_sum = hg1 + hg2
        
        # 2. Element-wise product (Symmetric) - captures nonlinear interactions
        hg_prod = hg1 * hg2
        
        # 3. Absolute difference (Symmetric) - captures differences
        hg_diff = torch.abs(hg1 - hg2)
        
        # Concatenate three symmetric features: [Sum, Prod, Diff] -> dimension 3 * nhid
        # Keep the same dimensionality as the original model (3 * nhid), no FC-layer change needed
        hg = torch.cat([hg_sum, hg_prod, hg_diff], dim=-1)
        
        # Clear intermediate variables
        del hg1, hg2, hg_sum, hg_prod, hg_diff

        # Step 7: Fully connected prediction
        h = nn.functional.relu(self.fc1(hg))
        h = self.dropout(h)
        h = nn.functional.relu(self.fc2(h))
        return self.fc3(h)

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
            raise AttributeError("Data module is missing check_protein_exists interface; protein IDs cannot be validated")
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
                        print(f"⚠️ Invalid format at line {line_idx+1}, skipping: {line}")
                        invalid_count += 1

            print(f"   ✅ {file_path}: {valid_count} valid samples, {invalid_count} invalid samples")
            total_valid += valid_count
            total_invalid += invalid_count

        print(f"📊 Total: {total_valid} valid samples, {total_invalid} invalid samples")

        if len(self.samples) == 0:
            raise ValueError("No valid test samples found! Please check data file format and protein IDs.")

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

def _get_lazy_cache_path(model_name: str) -> Path:
    _ensure_dir(LAZY_CACHE_DIR)
    safe_name = model_name.replace('/', '_').replace(':', '_').replace('\\', '_')
    return LAZY_CACHE_DIR / f"{safe_name}_predictions.pkl"


def _load_lazy_cache(model_name: str):
    cache_path = _get_lazy_cache_path(model_name)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded from lazy cache: {cache_path}")
            return data
        except Exception as e:
            print(f"⚠️ Failed to load lazy cache: {e}")
    return None


def _save_lazy_cache(model_name: str, predictions: np.ndarray, labels: np.ndarray, results: dict):
    cache_path = _get_lazy_cache_path(model_name)
    try:
        _ensure_dir(LAZY_CACHE_DIR)
        data = {
            'predictions': predictions,
            'labels': labels,
            'results': results,
            'model_name': model_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Predictions cached to: {cache_path}")
    except Exception as e:
        print(f"⚠️ Failed to save lazy cache: {e}")


def evaluate_model(model, test_loader, device, model_name="Model", save_txt=True, use_cache=True):
    if use_cache:
        cached = _load_lazy_cache(model_name)
        if cached is not None:
            results = cached['results']
            print(f"📊 {model_name} evaluation results (from cache):")
            print(f"   AUPRC: {results['auprc']:.4f}")
            print(f"   AUROC: {results['auroc']:.4f}")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   F1 Score: {results['f1']:.4f}")
            return results, cached['predictions'], cached['labels']
    
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

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    if len(np.unique(all_labels)) < 2:
        print(f"⚠️ {model_name}: test set contains only one class, full evaluation metrics cannot be computed")
        return None, all_predictions, all_labels
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_predictions)
    auprc = auc(recall_curve, precision_curve)
    auroc = roc_auc_score(all_labels, all_predictions)
    predictions_binary = all_predictions > 0.5
    accuracy = accuracy_score(all_labels, predictions_binary)
    precision_val = precision_score(all_labels, predictions_binary, zero_division=0)
    recall_val = recall_score(all_labels, predictions_binary, zero_division=0)
    f1 = f1_score(all_labels, predictions_binary, zero_division=0)
    mcc = matthews_corrcoef(all_labels, predictions_binary)
    results = {
        'model_name': model_name,
        'auprc': float(auprc),
        'auroc': float(auroc),
        'accuracy': float(accuracy),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1': float(f1),
        'mcc': float(mcc),
        'num_samples': int(len(all_labels)),
        'num_positive': int(sum(all_labels)),
        'num_negative': int(len(all_labels) - sum(all_labels))
    }
    _save_lazy_cache(model_name, all_predictions, all_labels, results)
    if save_txt:
        save_model_results_to_txt(results, model_name)
    print(f"📊 {model_name} evaluation results:")
    print(f"   AUPRC: {auprc:.4f}")
    print(f"   AUROC: {auroc:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision_val:.4f}")
    print(f"   Recall: {recall_val:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   MCC: {mcc:.4f}")
    print(f"   Samples: {int(len(all_labels))} (positive: {int(sum(all_labels))}, negative: {int(len(all_labels) - sum(all_labels))})")
    return results, all_predictions, all_labels

def plot_roc_curves(model_data: dict, save_suffix: str = "auprc"):
    _ensure_dir(PLOT_OUTPUT_DIR)
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    fold_idx = 0
    for model_name, data in model_data.items():
        if f'_{save_suffix.upper()}' not in model_name.upper() and save_suffix.upper() not in model_name.upper():
            continue
        predictions = data['predictions']
        labels = data['labels']
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = roc_auc_score(labels, predictions)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        fold_num = "?"
        for part in model_name.split('_'):
            if 'Fold' in part or 'fold' in part:
                fold_num = part.replace('Fold', '').replace('fold', '')
                break
        plt.plot(fpr, tpr, color=colors[fold_idx % len(colors)], 
                 lw=3, alpha=0.8, 
                 label=f'Fold {fold_num} (AUC = {roc_auc:.4f})')
        fold_idx += 1
    
    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax = plt.gca()
    xticks = ax.get_xticks().tolist()
    xticks = [x for x in xticks if x != 0.0]
    plt.xticks(xticks, fontsize=20)
    plt.legend(loc='lower right', fontsize=20, frameon=False)
    plt.grid(True, alpha=0.3)
    save_path = PLOT_OUTPUT_DIR / f'1xg_roc_curves_best_{save_suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 ROC curve saved: {save_path}")


def plot_pr_curves(model_data: dict, save_suffix: str = "auprc"):
    _ensure_dir(PLOT_OUTPUT_DIR)
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    auprcs = []
    fold_idx = 0
    for model_name, data in model_data.items():
        if f'_{save_suffix.upper()}' not in model_name.upper() and save_suffix.upper() not in model_name.upper():
            continue
        predictions = data['predictions']
        labels = data['labels']
        precision_arr, recall_arr, _ = precision_recall_curve(labels, predictions)
        auprc_val = auc(recall_arr, precision_arr)
        sorted_indices = np.argsort(recall_arr)
        recall_sorted = recall_arr[sorted_indices]
        precision_sorted = precision_arr[sorted_indices]
        interp_precision = np.interp(mean_recall, recall_sorted, precision_sorted)
        precisions.append(interp_precision)
        auprcs.append(auprc_val)
        fold_num = "?"
        for part in model_name.split('_'):
            if 'Fold' in part or 'fold' in part:
                fold_num = part.replace('Fold', '').replace('fold', '')
                break
        
        plt.plot(recall_arr, precision_arr, color=colors[fold_idx % len(colors)], 
                 lw=3, alpha=0.8, 
                 label=f'Fold {fold_num} (AUC = {auprc_val:.4f})')
        fold_idx += 1
    if precisions:
        mean_precision = np.mean(precisions, axis=0)
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)
        std_precision = np.std(precisions, axis=0)
        precision_upper = np.minimum(mean_precision + std_precision, 1)
        precision_lower = np.maximum(mean_precision - std_precision, 0)
    all_labels = np.concatenate([data['labels'] for data in model_data.values()])
    pos_ratio = all_labels.mean() if len(all_labels) > 0 else 0.0
    if pos_ratio > 0:
        plt.hlines(pos_ratio, 0, 1, colors="gray", linestyles="dashed", label=f"Baseline={pos_ratio:.4f}")

    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Recall", fontsize=26)
    plt.ylabel("Precision", fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax = plt.gca()
    xticks = ax.get_xticks().tolist()
    xticks = [x for x in xticks if x != 0.0]
    plt.xticks(xticks, fontsize=20)
    plt.legend(loc="lower left", fontsize=20, frameon=False)
    plt.grid(True, alpha=0.3)
    
    save_path = PLOT_OUTPUT_DIR / f'1xg_pr_curves_best_{save_suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 PR curve saved: {save_path}")


def generate_all_plots(all_results: list, model_predictions: dict):
    print("\n" + "="*60)
    print("📊 Starting visualization generation...")
    print("="*60)
    _ensure_dir(PLOT_OUTPUT_DIR)
    auprc_data = {k: v for k, v in model_predictions.items() if 'AUPRC' in k.upper()}
    auroc_data = {k: v for k, v in model_predictions.items() if 'AUROC' in k.upper()}
    
    # 1. Plot ROC and PR curves for models saved by best AUPRC
    if auprc_data:
        plot_roc_curves(auprc_data, save_suffix="auprc")
        plot_pr_curves(auprc_data, save_suffix="auprc")
    
    # 2. Plot ROC and PR curves for models saved by best AUROC
    if auroc_data:
        plot_roc_curves(auroc_data, save_suffix="auroc")
        plot_pr_curves(auroc_data, save_suffix="auroc")
    
    print("\n✅ All visualizations generated!")
    print(f"📁 Plot output directory: {PLOT_OUTPUT_DIR}")

def save_model_results_to_txt(results, model_name):
    _ensure_dir(FANHUA_OUTPUT_DIR)
    filename = FANHUA_OUTPUT_DIR / f"generalization_test_{model_name.replace('/', '_').replace(':', '_')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Protein-Protein Interaction Prediction Model Generalization Test Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model Name: {results['model_name']}\n")
        f.write(f"Test Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Evaluation Metrics:\n")
        f.write(f"{'='*30}\n")
        f.write(f"AUPRC (Area under Precision-Recall Curve): {results['auprc']:.6f}\n")
        f.write(f"AUROC (Area under ROC Curve):              {results['auroc']:.6f}\n")
        f.write(f"Accuracy:                                  {results['accuracy']:.6f}\n")
        f.write(f"Precision:                                 {results['precision']:.6f}\n")
        f.write(f"Recall:                                    {results['recall']:.6f}\n")
        f.write(f"F1 Score:                                  {results['f1']:.6f}\n")
        f.write(f"\n")
        f.write(f"Sample Statistics:\n")
        f.write(f"{'='*30}\n")
        f.write(f"Total Samples:    {results['num_samples']}\n")
        f.write(f"Positive Samples: {results['num_positive']}\n")
        f.write(f"Negative Samples: {results['num_negative']}\n")
        f.write(f"Positive Ratio:   {results['num_positive']/results['num_samples']:.4f}\n")
        f.write(f"Negative Ratio:   {results['num_negative']/results['num_samples']:.4f}\n")
        f.write(f"\n")
        f.write(f"Metric Descriptions:\n")
        f.write(f"{'='*30}\n")
        f.write(f"AUPRC: Suitable for imbalanced datasets; higher is better\n")
        f.write(f"AUROC: Overall classification performance metric; higher is better\n")
        f.write(f"Accuracy: Ratio of correctly predicted samples to all samples\n")
        f.write(f"Precision: Ratio of true positives among predicted positives\n")
        f.write(f"Recall: Ratio of correctly predicted positives among actual positives\n")
        f.write(f"F1 Score: Harmonic mean of precision and recall\n")

    print(f"💾 {model_name} results saved to: {filename}")

def test_all_models(data_files, device=DEFAULT_DEVICE, batch_size=DEFAULT_BATCH_SIZE, use_cache=True):
    print("🚀 Starting generalization capability test...")
    _ensure_dir(FANHUA_OUTPUT_DIR)
    _ensure_dir(LAZY_CACHE_DIR)
    _ensure_dir(PLOT_OUTPUT_DIR)
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
                print("Preloading protein coordinates into cache (memory/disk)...")
                data_module.init_coord_cache() 
                data_module.batch_preload_coordinates(all_ids, max_workers=4, label="fanhua-preload", every=2000)
    except Exception as _e:
        print(f"⚠️ Coordinate preloading failed (continuing): {_e}")
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

    model_paths = resolve_model_paths()
    if not model_paths:
        print("⚠️ No evaluable model paths found; please check FANHUA_MODEL_PATHS / FANHUA_MODEL_DIR settings")
        return [], {}

    all_cached = True
    if use_cache:
        for model_name in model_paths.keys():
            cache_path = _get_lazy_cache_path(model_name)
            if not cache_path.exists():
                all_cached = False
                break
    else:
        all_cached = False

    in_dim = None
    if not all_cached:
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
            raise ValueError("Unable to obtain valid samples from test data to infer input dimension")

        in_dim = sample_batch[0].ndata['fea'].shape[1]
        print(f"📏 Input feature dimension: {in_dim}")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=False
        )

    all_results = []
    model_predictions = {} 

    for model_name, model_path in model_paths.items():
        if use_cache:
            cached = _load_lazy_cache(model_name)
            if cached is not None:
                results = cached['results']
                predictions = cached['predictions']
                labels = cached['labels']
                all_results.append(results)
                model_predictions[model_name] = {
                    'predictions': predictions,
                    'labels': labels
                }
                print(f"📊 {model_name} (from cache) - AUPRC: {results['auprc']:.4f}, AUROC: {results['auroc']:.4f}")
                continue
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model file does not exist: {model_path}")
            continue

        if in_dim is None:
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
            if sample_batch is None:
                raise ValueError("Unable to obtain valid samples")
            in_dim = sample_batch[0].ndata['fea'].shape[1]
            print(f"📏 Input feature dimension: {in_dim}")
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0, pin_memory=False
            )

        print(f"\n🔄 Loading model: {model_name}")
        model = MyGCN(in_dim=in_dim, nhid=nhid, dropout=dropout, time_steps=time_steps)
        ok = _load_model_weights(model, model_path, device)
        if not ok:
            continue
        model = model.to(device)
        print(f"✅ Model loaded successfully: {model_name}")
        eval_result = evaluate_model(model, test_loader, device, model_name, save_txt=True, use_cache=False)
        if eval_result[0] is not None:
            results, predictions, labels = eval_result
            all_results.append(results)
            model_predictions[model_name] = {
                'predictions': predictions,
                'labels': labels
            }
        del model
        torch.cuda.empty_cache()
        gc.collect()

    if hasattr(data_module, "clear_graph_cache"):
        data_module.clear_graph_cache()

    return all_results, model_predictions

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
    print("\n📊 Generalization test summary:")
    print("="*80)
    print(f"{'Model Name':<20} {'AUPRC':<8} {'AUROC':<8} {'Accuracy':<8} {'F1':<8} {'Samples':<8}")
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
    use_cache = os.getenv("FANHUA_USE_CACHE", "1") == "1"
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️ CUDA unavailable, switching to CPU")
        device = 'cpu'

    print(f" Using device: {device}")
    print(f" Lazy cache: {'enabled' if use_cache else 'disabled'}")
    print(f" Lazy cache directory: {LAZY_CACHE_DIR}")
    print(f" Plot output directory: {PLOT_OUTPUT_DIR}")
    neg_files = [p for p, lab in data_files if lab == 0]
    pos_files = [p for p, lab in data_files if lab == 1]
    print(f" Negative sample files: {', '.join(neg_files) if neg_files else 'None'}")
    print(f" Positive sample files: {', '.join(pos_files) if pos_files else 'None'}")
    print(f" Batch size: {batch_size}")

    try:
        results, model_predictions = test_all_models(data_files, device, batch_size, use_cache=use_cache)
        if not results:
            print("❌ No model was successfully evaluated")
            return
        save_results(results, output_file)
        if model_predictions:
            generate_all_plots(results, model_predictions)
        else:
            print("⚠️ No prediction data, skipping plotting")

        print("\n Generalization capability test completed!")
        print(f" Result directory: {FANHUA_OUTPUT_DIR}")
        print(f" Plot directory: {PLOT_OUTPUT_DIR}")
        print(f" Cache directory: {LAZY_CACHE_DIR}")

    except Exception as e:
        print(f"❌ Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
