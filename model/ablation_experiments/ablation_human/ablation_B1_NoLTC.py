import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import sys
import time
import json
import argparse
import traceback
import socket
import math
import random
import glob
import pickle
import threading
from pathlib import Path
from collections import defaultdict
from typing import Optional
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

import sklearn.metrics as metrics
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt

os.environ['PPI_OUTPUT_DIR'] = "LNGCN_main/results/ablation/ablation_human/B1_NoLTC"
os.environ['PPI_COORD_CACHE_SIZE'] = '20480' 
os.environ['PPI_GRAPH_CACHE_SIZE'] = '2048'  
os.environ['PPI_DATALOADER_WORKERS'] = '4' 
os.environ['PPI_QUIET'] = '0' 
os.environ['PPI_ENABLE_COORD_PRELOAD'] = '1' 
os.environ['PPI_COORD_PRELOAD_THREADS'] = '4'
os.environ['PPI_STRICT_COORD_MATCH'] = '0' 

STRICT_TIMESTEP = os.getenv("PPI_STRICT_TIMESTEP", "1") == "1"
DDP_DEBUG = os.getenv("PPI_DDP_DEBUG", "0") == "1"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True  
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import dgl
from dgl.nn import GraphConv, Set2Set
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP, FullyConnected, Wiring

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
data_module = importlib.import_module('B1_data')
try:
    from ablation_models_v1 import AblationGCN
except ImportError as e:
    print(f"❌ Model import failed: {e}")
    print("Please ensure ablation_models_v1.py exists and contains the AblationGCN class")
    sys.exit(1)

SEED = 867482
def setup_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

OUTPUT_ROOT = Path(data_module._resolve_output_root())

def ensure_output_dirs():
    """Ensure the output directory exists."""
    try:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def p(*parts) -> str:
    return str(OUTPUT_ROOT.joinpath(*parts))

EPOCHS = 50
LR = 0.0001
nhid = 256
nhidh = 128
nhidhh = 64
dropout = 0.4
time_steps = 5
ode_unfolds = 2
BATCH_SIZE_SAMPLES = 500  
SAMPLES_BATCH_DIR = OUTPUT_ROOT / "18844all-samples_batches"
MAX_WORKERS = min(8, os.cpu_count())  

TRAIN_BATCH_SIZE = int(os.getenv("PPI_TRAIN_BATCH_SIZE", "80"))
VAL_BATCH_SIZE = int(os.getenv("PPI_VAL_BATCH_SIZE", "180"))
TEST_BATCH_SIZE = int(os.getenv("PPI_TEST_BATCH_SIZE", "256"))

EARLY_STOP_MIN_EPOCHS = int(os.getenv("PPI_EARLY_STOP_MIN_EPOCHS", "15"))
EARLY_STOP_PATIENCE = int(os.getenv("PPI_EARLY_STOP_PATIENCE", "15"))
EARLY_STOP_TRAIN_LOSS_THRESHOLD = float(os.getenv("PPI_EARLY_STOP_TRAIN_LOSS_THRESHOLD", "0.001")) 
EARLY_STOP_VAL_LOSS_PATIENCE = int(os.getenv("PPI_EARLY_STOP_VAL_LOSS_PATIENCE", "3"))
COMBINED_METRIC_WEIGHTS = {
    'auroc': 0.35,
    'auprc': 0.30,
    'f1': 0.15,
    'accuracy': 0.05,
    'mcc': 0.15,
}

DEBUG_SHAPES = os.getenv("PPI_DEBUG_SHAPES", "0") == "1"

class CfCCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, is_first_layer=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_first_layer = is_first_layer
        input_concat_dim = input_dim if is_first_layer else (input_dim + hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_concat_dim, hidden_dim),
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
            timestep_feat = feat[:, -1:] 
        else:
            base_feat = feat
            timestep_feat = torch.zeros(feat.size(0), 1, device=feat.device, dtype=feat.dtype)

        h = torch.zeros(base_feat.size(0), self.hidden_dim, device=feat.device, dtype=feat.dtype)
        x_in = base_feat
        for i, cell in enumerate(self.cells):
            h = cell(h, x_in, t=1.0)
            x_in = h

        output = torch.cat([h, timestep_feat], dim=1)
        return output

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
        return numerator / denominator

    def forward(self, t, x, dt=0.01):
        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            f = self.off_compute_dynamics(x, t)
            x = self.off_ode_step(x, f, delta_t)
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
            x = self.ltc(t=step * dt, x=x, dt=dt)
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
        return

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
            distance_times = features[:, -1:]
            x = features[:, :-1]
        else:
            distance_times = torch.zeros(features.size(0), 1, device=features.device, dtype=features.dtype)
            x = features

        if DEBUG_SHAPES:
            print(f"EnhancedDistanceLTC input - features: {features.shape}, x: {x.shape}, distance_times: {distance_times.shape}")

        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            gate = self._compute_enhanced_gating(graph, x, distance_times)
            f = self.transform(gate * x)
            x = self._ode_step(x, f, delta_t)

        return self.out(x)

class EnhancedLTCLayer(nn.Module):
    def __init__(self, nhid, time_steps=5, residual=True):
        super().__init__()
        self.enhanced_ltc = EnhancedDistanceLTC(nhid=nhid, ode_unfolds=ode_unfolds)
        self.graph_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        self.multi_scale_conv = nn.ModuleList([
            dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True),  # 1-hop
            dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True),  # 2-hop
        ])
        self.scale_fusion = nn.Linear(nhid * 3, nhid) 
        self.bn = nn.LayerNorm(nhid)
        self.relu = nn.ReLU()
        self.residual = residual

        nn.init.xavier_normal_(self.graph_conv.weight)
        for conv in self.multi_scale_conv:
            nn.init.xavier_normal_(conv.weight)
        nn.init.xavier_normal_(self.scale_fusion.weight)

    def forward(self, graph, feat):
        ltc_feat = self.enhanced_ltc(graph, feat) 
        residual = ltc_feat
        graph_feat = self.graph_conv(graph, ltc_feat)
        scale1_feat = self.multi_scale_conv[0](graph, ltc_feat)
        scale2_feat = self.multi_scale_conv[1](graph, scale1_feat)
        multi_scale = torch.cat([ltc_feat, scale1_feat, scale2_feat], dim=-1)
        fused_feat = self.scale_fusion(multi_scale)

        x = self.bn(graph_feat + fused_feat)
        x = self.relu(x)

        if self.residual:
            x = x + residual
        return x

class LTC(nn.Module):
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

    def _compute_hybrid_gating(self, graph, x, t):
        t_broadcast = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype) * t
        x_t = torch.cat([x, t_broadcast], dim=1)
        gate = self.w_gate(x_t)
        hidden = self.w_hid(x_t)
        dynamic = self.transform(gate * hidden)
        return dynamic

    def _compute_dynamics(self, graph, x, t):
        return self._compute_hybrid_gating(graph, x, t)

    def _ode_step(self, x, f, delta_t):
        tau = torch.relu(self.tau) + 1e-8
        tau_inv = 1.0 / tau
        numerator = x + delta_t * f * self.A
        denominator = 1 + delta_t * (tau_inv + f)
        return numerator / denominator

    def forward(self, graph, x, t=0.0, dt=0.01):
        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            f = self._compute_dynamics(graph, x, t)
            x = self._ode_step(x, f, delta_t)
        return self.out(x)

BaselinePPIModel = AblationGCN


def update_batch_sizes(train: Optional[int] = None, val: Optional[int] = None, test: Optional[int] = None):
    global TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE

    if train is not None:
        TRAIN_BATCH_SIZE = max(1, int(train))
        os.environ["PPI_TRAIN_BATCH_SIZE"] = str(TRAIN_BATCH_SIZE)
    if val is not None:
        VAL_BATCH_SIZE = max(1, int(val))
        os.environ["PPI_VAL_BATCH_SIZE"] = str(VAL_BATCH_SIZE)
    if test is not None:
        TEST_BATCH_SIZE = max(1, int(test))
        os.environ["PPI_TEST_BATCH_SIZE"] = str(TEST_BATCH_SIZE)

    return TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE


def check_disk_space(path, required_gb=1.0):
    try:
        stat = os.statvfs(path)
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return available_gb >= required_gb
    except Exception:
        return True 


def format_metric_dict(metrics: Optional[dict]) -> Optional[dict]:
    if not metrics:
        return None
    formatted = {}
    for key, value in metrics.items():
        if key == 'epoch' and value is not None:
            formatted[key] = int(value)
        else:
            formatted[key] = sanitize_metric(value)
    return formatted

DATALOADER_WORKERS = int(os.getenv("PPI_DATALOADER_WORKERS", "4"))
PREFETCH_FACTOR = int(os.getenv("PPI_PREFETCH_FACTOR", "2"))
PIN_MEMORY = os.getenv("PPI_PIN_MEMORY", "1") == "1"
PERSISTENT_WORKERS = os.getenv("PPI_PERSISTENT_WORKERS", "0") == "1"
BASE_PORT = int(os.getenv("PPI_BASE_PORT", "27519"))
INPUT_DATA_CONFIG = {
    'split_mode': 1, 
    'predefined_split_files': {
        'train': "LNGCN_main/data/ablation_experiments/fold_5_train.txt",
        'val': "LNGCN_main/data/ablation_experiments/fold_5_val.txt",
        'test': "LNGCN_main/data/ablation_experiments/fold_5_test.txt",
    },
}

def find_free_port(preferred: int = None, max_tries: int = 32) -> int:
    def _is_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    if preferred is not None:
        for offset in range(max_tries):
            port = preferred + offset
            if _is_free(port):
                return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        sock.listen(1)
        return sock.getsockname()[1]

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            current_device = torch.cuda.current_device()
            torch.cuda.empty_cache()
        except:
            pass

def get_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return "GPU Memory - Allocated: " + f"{allocated:.2f}" + "GB, Cached: " + f"{cached:.2f}" + "GB"
    return "CUDA not available"

def sanitize_metric(value):
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return value

def gather_variable_tensor(tensor: torch.Tensor, world_size: int, device: torch.device):
    if tensor is None:
        tensor = torch.empty(0, device=device)
    if not dist.is_available() or not dist.is_initialized() or world_size == 1:
        return [tensor]

    tensor = tensor.contiguous()
    if tensor.ndim == 0:
        tensor = tensor.view(1)

    length_tensor = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    gathered_lengths = [torch.zeros_like(length_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, length_tensor)

    max_length = int(max(int(l.item()) for l in gathered_lengths))
    if tensor.shape[0] < max_length:
        pad_shape = (max_length - tensor.shape[0],) + tensor.shape[1:]
        padding = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)

    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)

    results = []
    for gathered_tensor, length in zip(gather_list, gathered_lengths):
        valid_length = int(length.item())
        results.append(gathered_tensor[:valid_length].clone())
    return results

class LazySampleDataset(Dataset):
    def __init__(self, batch_dir, indices=None):
        self.batch_dir = batch_dir
        self.batch_files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.pkl")))
        self.batch_sizes = []
        self.cumulative_sizes = [0]
        total_samples = 0

        for batch_file in self.batch_files:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                batch_size = len(batch_data)
                self.batch_sizes.append(batch_size)
                total_samples += batch_size
                self.cumulative_sizes.append(total_samples)

        if indices is not None:
            self.indices = np.array(indices)
        else:
            self.indices = np.arange(total_samples)

        self.cache = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        batch_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:], 1):
            if global_idx < cum_size:
                batch_idx = i - 1
                break

        batch_start = self.cumulative_sizes[batch_idx]
        local_idx = global_idx - batch_start

        batch_file = self.batch_files[batch_idx]
        if batch_file not in self.cache:
            with open(batch_file, 'rb') as f:
                self.cache[batch_file] = pickle.load(f)

        batch_data = self.cache[batch_file]
        if local_idx >= len(batch_data):
            raise IndexError(f"Index {local_idx} out of range for batch {batch_idx} (batch size: {len(batch_data)})")

        return batch_data[local_idx]

    def clear_cache(self):
        self.cache.clear()

class EmptyDataset(Dataset):
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError("Empty dataset")

def load_predefined_splits():
    split_files = INPUT_DATA_CONFIG['predefined_split_files']
    
    train_file = Path(split_files['train'])
    val_file = Path(split_files['val'])
    test_file = Path(split_files['test'])
    
    for f in [train_file, val_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(f"Data file not found: {f}")
    
    def load_samples(file_path):
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    samples.append((parts[0], parts[1], int(parts[2])))
        return samples
    
    print(f"📂 Loading predefined splits...")
    train_samples = load_samples(train_file)
    val_samples = load_samples(val_file)
    test_samples = load_samples(test_file)
    
    print(f"✅ Load complete - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples

def build_samples_from_list(samples, batch_dir, rank=0):
    os.makedirs(batch_dir, exist_ok=True)

    existing_batches = glob.glob(os.path.join(batch_dir, "batch_*.pkl"))
    if len(existing_batches) > 0:
        print(f"🧹 Cleaning up {len(existing_batches)} existing old batch files")
        for batch_file in existing_batches:
            try:
                os.remove(batch_file)
            except Exception as e:
                print(f"⚠️ Failed to delete file {batch_file}: {e}")

    all_protein_ids = set()
    for p1, p2, _ in samples:
        all_protein_ids.update([p1, p2])
    
    total_unique = len(all_protein_ids)
    print(f"📊 Found {total_unique} unique proteins, starting preload...")

    data_module.batch_preload_coordinates(list(all_protein_ids), max_workers=4, label="preload", quiet=(rank != 0))

    successful, failed = 0, 0
    batch_idx, current = 0, []
    lock = threading.Lock()

    def worker(p1, p2, lbl):
        try:
            if not data_module.check_protein_exists_fast(p1): raise ValueError(f"Protein {p1} not found")
            if not data_module.check_protein_exists_fast(p2): raise ValueError(f"Protein {p2} not found")
            return True, (p1, p2, lbl)
        except Exception as e:
            try:
                data_module.append_fail_cache(f"{p1},{p2}", f"pair-check: {e}", label="pairs")
            except Exception:
                pass
            return False, None

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    for i in range(0, len(samples), 5000): 
        batch_samples = samples[i:i+5000]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = {exe.submit(worker, p1, p2, lbl): (p1, p2) for p1, p2, lbl in batch_samples}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batch {i//5000}"):
                ok, sample = fut.result()
                with lock:
                    if ok:
                        current.append(sample); successful += 1
                        if len(current) >= BATCH_SIZE_SAMPLES:
                            save_path = os.path.join(batch_dir, f"batch_{batch_idx}.pkl")
                            with open(save_path, "wb") as f: pickle.dump(current, f)
                            batch_idx += 1; current = []
                    else:
                        failed += 1

    if current:
        save_path = os.path.join(batch_dir, f"batch_{batch_idx}.pkl")
        with open(save_path, "wb") as f: pickle.dump(current, f)

    del current
    gc.collect()
    print(f"Samples built: {successful} ok, {failed} failed")

BASE_CONFIG = {
    'description': 'B1_NoLTC: Fully remove LTC temporal processing and keep only graph convolution',
    'scientific_hypothesis': 'Validate the importance of LTC temporal modeling for protein-protein interaction prediction',

    'use_cfc': True,
    'cfc_type': 'original',
    'cfc_layers': 3,

    'use_ltc': False,
    'ltc_residual': False,
    'use_distance': False,

    'use_structure_enhance': True,
    'use_multi_scale': True,

    'pool_type': 'set2set',  
    
    'fusion_type': 'symmetric',  
    
    'use_ltc_dense': True,
    'fc_layers': 2,
    'dropout': dropout,
}

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(self.alpha, device=targets.device, dtype=torch.float)
        alpha_weights = self.alpha[targets]

        focal_loss = alpha_weights * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / self.warmup_steps
            return [self.max_lr * progress for group in self.optimizer.param_groups]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_progress = (1 + np.cos(np.pi * progress)) / 2
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_progress for group in
                    self.optimizer.param_groups]

try:
    from ablation_models_v1 import AblationGCN
except ImportError as e:
    if os.getenv('PPI_QUIET', '0') != '1':
        print(f"❌ Model import failed: {e}")
        print("Please ensure ablation_models_v1.py exists and contains the AblationGCN class")
    sys.exit(1)

def setup(rank, world_size, port):
    """Set up the distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    os.environ['NCCL_DEBUG'] = os.environ.get('NCCL_DEBUG', 'WARN')
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_SHM_DISABLE'] = '0'
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = os.environ.get('TORCH_NCCL_BLOCKING_WAIT', '1')
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ['TORCH_NCCL_ENABLE_MONITORING'] = os.environ.get('TORCH_NCCL_ENABLE_MONITORING', '1')
    os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = os.environ.get('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '1800')
    if 'NCCL_SOCKET_IFNAME' in os.environ:
        del os.environ['NCCL_SOCKET_IFNAME']
    if 'CUDA_LAUNCH_BLOCKING' in os.environ:
        del os.environ['CUDA_LAUNCH_BLOCKING']

    from datetime import timedelta
    
    if rank >= world_size:
        raise RuntimeError(f"Rank {rank} exceeds world_size {world_size}")
    
    try:
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: Successfully set GPU device {rank}")
    except Exception as e:
        raise RuntimeError(f"Rank {rank}: Unable to set GPU device {rank} - {e}")
        print(f"Rank {rank}: Failed to set GPU device: {e}")
        raise
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=3600)
    )

def cleanup():
    try:
        dist.destroy_process_group()
    except Exception:
        pass

def train(rank, world_size, port, train_idx, val_idx, test_idx, experiment_config, exp_name):
    exp_output_dir = OUTPUT_ROOT
    setup(rank, world_size, port)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        raise

    if rank == 0:
        print()
        print("="*60)
        print("🚀 Starting ablation experiment: " + exp_name)
        print("="*60)
        print("Experiment description: " + experiment_config.get('description', 'N/A'))
        print("Scientific hypothesis: " + experiment_config.get('scientific_hypothesis', 'N/A'))
        print("Configuration details: " + str({k: v for k, v in experiment_config.items() if k not in ['description', 'scientific_hypothesis']}))
        print("Training device: cuda:" + str(rank))
        print("Memory status: " + get_memory_usage())
        print("Output directory: " + str(exp_output_dir))
        print("="*60)
        print()
    
    graph_cache_size = int(os.getenv("PPI_GRAPH_CACHE_SIZE", "2048"))
    coord_cache_size = int(os.getenv("PPI_COORD_CACHE_SIZE", "20480"))
    data_module.init_graph_cache(max_size=graph_cache_size)
    data_module.init_coord_cache(max_size=coord_cache_size)
    if rank == 0:
        print(f"Graph cache limit: {graph_cache_size}, Coordinate cache limit: {coord_cache_size}")

    if rank == 0:
        print("🚀 Starting to preload protein coordinates for training data...")
    
    all_relevant_proteins = set()
    for idx in train_idx + val_idx + test_idx:
        try:
            p1, p2, _ = train_dataset.samples[idx] if hasattr(train_dataset, 'samples') else train_dataset[idx]
            all_relevant_proteins.update([p1, p2])
        except:
            pass
    
    data_module.batch_preload_coordinates(
        list(all_relevant_proteins), 
        max_workers=int(os.getenv("PPI_COORD_PRELOAD_THREADS", "4")), 
        label=f"train-fold{rank}", 
        quiet=(rank != 0)
    )
    
    if rank == 0:
        print("✅ Coordinate preload complete")

    try:
        train_dataset = LazySampleDataset(SAMPLES_BATCH_DIR, train_idx)
        val_dataset = LazySampleDataset(SAMPLES_BATCH_DIR, val_idx)
        test_dataset = LazySampleDataset(SAMPLES_BATCH_DIR, test_idx)
        if len(train_dataset) == 0:
            raise ValueError("Training set is empty, cannot run training")
        if rank == 0:
            print("✅ Dataset preparation complete")
            print("   Train set: " + str(len(train_dataset)) + " samples")
            print("   Val set: " + str(len(val_dataset)) + " samples")
            print("   Test set: " + str(len(test_dataset)) + " samples")
    except Exception as e:
        print(f"Dataset preparation failed: {e}")
        raise

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    per_gpu_train_batch = max(1, TRAIN_BATCH_SIZE)
    per_gpu_val_batch = max(1, VAL_BATCH_SIZE)
    per_gpu_test_batch = max(1, TEST_BATCH_SIZE)

    if rank == 0:
        print(f"Batch settings: per-GPU train batch={per_gpu_train_batch}, global effective batch={per_gpu_train_batch * world_size}")
        print(f"                val batch={per_gpu_val_batch}, test batch={per_gpu_test_batch}")

    safe_num_workers = min(DATALOADER_WORKERS, os.cpu_count() or 1)
    safe_pin_memory = PIN_MEMORY and torch.cuda.is_available()

    effective_prefetch = PREFETCH_FACTOR if (safe_num_workers > 0 and PREFETCH_FACTOR > 0) else None
    persistent_flag = PERSISTENT_WORKERS and safe_num_workers > 0

    if rank == 0:
        print(f"🔧 Multi-GPU safety config: num_workers={safe_num_workers}, pin_memory={safe_pin_memory}, prefetch={effective_prefetch}, "
              f"train_batch={per_gpu_train_batch}")

    loader_common_kwargs = dict(
        pin_memory=safe_pin_memory,
        num_workers=safe_num_workers,
        persistent_workers=persistent_flag
    )
    if safe_num_workers > 0 and effective_prefetch is not None:
        loader_common_kwargs['prefetch_factor'] = effective_prefetch
        loader_common_kwargs['multiprocessing_context'] = torch.multiprocessing.get_context('spawn')  # Fix worker abort issue

    collate_fn = data_module.collate_GCN

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_train_batch,
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True,
        **loader_common_kwargs
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_val_batch,
        sampler=val_sampler,
        collate_fn=collate_fn,
        **loader_common_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=per_gpu_test_batch,
        collate_fn=collate_fn,
        **loader_common_kwargs
    )

    first_valid_batch = None
    for idx in range(len(train_dataset)):
        candidate = collate_fn([train_dataset[idx]])
        if candidate is not None:
            first_valid_batch = candidate
            break
    if first_valid_batch is None:
        raise RuntimeError("No valid samples in training set; cannot initialize model")

    batch_g1, batch_g2, batch_labels = first_valid_batch
    nfeat = batch_g1.ndata['fea'].shape[1]
    del batch_g1, batch_g2, batch_labels, first_valid_batch
    torch.cuda.empty_cache()

    if rank == 0:
        print("📊 Input feature dimension: " + str(nfeat))

    model = AblationGCN(
        in_dim=nfeat,
        nhid=nhid,
        dropout=dropout,
        time_steps=time_steps,
        experiment_config=experiment_config
    ).to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if rank == 0:
        train_labels = []
        sample_size = min(10000, len(train_dataset))
        for i in range(sample_size):
            _, _, label = train_dataset[i]
            train_labels.append(label)

        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.tensor(class_weights, device=device, dtype=torch.float)
    else:
        class_weights = torch.tensor([1.0, 1.0], device=device, dtype=torch.float)
    loss_func = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    max_grad_norm = 1.0
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        max_lr=LR,
        min_lr=LR / 100
    )
    if rank == 0:
        print(f"✅ Ablation model imported successfully")
        print(f"✅ Model initialization complete - based on v9.26")
        print(f"   Parameter count: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Loss function: FocalLoss (alpha={class_weights.tolist()}, gamma=2)")
        print(f"   Optimizer: AdamW (lr={LR}, weight_decay=1e-4)")
        print(f"   Scheduler: WarmupCosineScheduler (warmup_steps={warmup_steps})")
        print(f"   Gradient clipping: max_norm={max_grad_norm}")

    best_auprc = 0.0
    best_auroc = 0.0
    best_epoch_auprc = 0
    best_epoch_auroc = 0
    best_val_metrics = {}
    best_train_metrics = {}
    best_combined_score = float('-inf')
    best_combined_epoch = None
    no_improve_epochs = 0
    early_stop_triggered = False
    early_stop_epoch = None
    early_stop_status = ""
    best_val_loss = float('inf')
    val_loss_no_improve_epochs = 0
    current_train_metrics = {}
    current_val_metrics = {}
    current_combined_score = float('nan')

    log_file = exp_output_dir / f"{exp_name}_training_log.txt"
    timing_log_file = exp_output_dir / f"{exp_name}_timing_log.txt"
    epoch_timing_file = exp_output_dir / f"{exp_name}_epoch_timing.txt"
    best_auprc_model = exp_output_dir / f"{exp_name}_best_auprc.pt"
    best_auroc_model = exp_output_dir / f"{exp_name}_best_auroc.pt"
    results_file = exp_output_dir / f"{exp_name}_results.json"
    test_metrics_auprc_file = exp_output_dir / f"{exp_name}_test_metrics_auprc.txt"
    test_metrics_auroc_file = exp_output_dir / f"{exp_name}_test_metrics_auroc.txt"

    if rank == 0:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Ablation experiment training log: {exp_name}\n")
            f.write(f"Experiment description: {experiment_config.get('description', 'N/A')}\n")
            f.write(f"Scientific hypothesis: {experiment_config.get('scientific_hypothesis', 'N/A')}\n")
            f.write(f"Configuration details: {str({k: v for k, v in experiment_config.items() if k not in ['description', 'scientific_hypothesis']})}\n")
            f.write(f"Training parameters: epochs={EPOCHS}, batch_size={TRAIN_BATCH_SIZE}, lr={LR}\n")
            f.write(f"{'='*80}\n")

        with open(timing_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Ablation experiment timing log: {exp_name}\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")

        with open(epoch_timing_file, 'w', encoding='utf-8') as f:
            f.write("Epoch\tEpoch_Time(s)\tEpoch_Time(min)\tTrain_Time(s)\tVal_Time(s)\tCumulative_Time(s)\n")

    start_time = time.time()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            train_sampler.set_epoch(epoch)
            
            train_phase_start = time.time()
            model.train()
            total_train_loss = 0.0
            train_preds = []  
            train_labels_list = [] 
            processed_batches = 0
            
            if rank == 0:
                print()
                print(f"Epoch {epoch+1}/{EPOCHS}")
                print("-" * 60)
            
            for batch_idx, batch in enumerate(train_loader):
                if batch is None:
                    continue
                g1, g2, labels = batch
                g1, g2 = g1.to(device), g2.to(device)
                labels = labels.long().to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if device.type == 'cuda' else torch.cuda.amp.autocast(enabled=False):
                    logits = model(g1, g2)
                    logits = torch.clamp(logits, min=-20.0, max=20.0)
                    loss = loss_func(logits, labels)

                nan_local = torch.tensor([
                    int(
                        torch.isnan(loss) or torch.isinf(loss) or
                        torch.isnan(logits).any() or torch.isinf(logits).any()
                    )
                ], device=device, dtype=torch.int32)
                if dist.is_available() and dist.is_initialized() and world_size > 1:
                    dist.all_reduce(nan_local, op=dist.ReduceOp.MAX)
                if nan_local.item() > 0:
                    if rank == 0 and batch_idx % 10 == 0:  # Reduce print frequency
                        print(f"⚠️ Epoch {epoch+1} Batch {batch_idx+1}: NaN/Inf detected, skipping this step")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                grad_nan = torch.tensor([
                    int(torch.isnan(grad_norm) or torch.isinf(grad_norm))
                ], device=device, dtype=torch.int32)
                if dist.is_available() and dist.is_initialized() and world_size > 1:
                    dist.all_reduce(grad_nan, op=dist.ReduceOp.MAX)
                if grad_nan.item() > 0:
                    if rank == 0 and batch_idx % 10 == 0:  # Reduce print frequency
                        print(f"⚠️ Epoch {epoch+1} Batch {batch_idx+1}: Abnormal gradient detected, skipping this step")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_train_loss += loss.item()
                processed_batches += 1
                
                if logits.ndim == 2 and logits.shape[1] >= 2:
                    probs = torch.softmax(logits.detach(), dim=-1)[:, 1]
                else:
                    probs = torch.sigmoid(logits.detach().view(-1))
                train_preds.append(probs) 
                train_labels_list.append(labels)  
                
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            avg_train_loss = total_train_loss / processed_batches if processed_batches > 0 else float('nan')
            train_phase_time = time.time() - train_phase_start
            
            if train_preds:
                train_pred_tensor = torch.cat(train_preds) 
                train_label_tensor = torch.cat(train_labels_list) 
            else:
                train_pred_tensor = torch.empty(0, device=device, dtype=torch.float32)
                train_label_tensor = torch.empty(0, device=device, dtype=torch.long)

            gathered_train_preds = gather_variable_tensor(train_pred_tensor, world_size, device)
            gathered_train_labels = gather_variable_tensor(train_label_tensor, world_size, device)

            if rank == 0:
                train_scores = torch.cat(gathered_train_preds).cpu().to(torch.float32).numpy() if gathered_train_preds else np.array([])
                train_labels_all = torch.cat(gathered_train_labels).cpu().to(torch.float32).numpy() if gathered_train_labels else np.array([])
                
                if len(train_scores) > 0 and len(np.unique(train_labels_all)) >= 2:
                    train_auprc = metrics.average_precision_score(train_labels_all, train_scores)
                    train_auroc = metrics.roc_auc_score(train_labels_all, train_scores)
                    train_preds_binary = (train_scores > 0.5).astype(int)
                    train_precision = metrics.precision_score(train_labels_all, train_preds_binary)
                    train_recall = metrics.recall_score(train_labels_all, train_preds_binary)
                    train_f1 = metrics.f1_score(train_labels_all, train_preds_binary)
                    train_acc = metrics.accuracy_score(train_labels_all, train_preds_binary)
                    train_mcc = metrics.matthews_corrcoef(train_labels_all, train_preds_binary)
                else:
                    train_auprc = float('nan')
                    train_auroc = float('nan')
                    train_precision = float('nan')
                    train_recall = float('nan')
                    train_f1 = float('nan')
                    train_acc = float('nan')
                    train_mcc = float('nan')
            else:
                train_auprc = float('nan')
                train_auroc = float('nan')
                train_precision = float('nan')
                train_recall = float('nan')
                train_f1 = float('nan')
                train_acc = float('nan')
                train_mcc = float('nan')
            
            val_phase_start = time.time()
            model.eval()
            total_val_loss = 0.0
            val_preds = []
            val_labels_list = []
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    g1, g2, labels = batch
                    g1, g2 = g1.to(device), g2.to(device)
                    labels = labels.long().to(device)

                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if device.type == 'cuda' else torch.cuda.amp.autocast(enabled=False):
                        logits = model(g1, g2)
                        logits = torch.clamp(logits, min=-20.0, max=20.0)
                        loss = loss_func(logits, labels)

                    total_val_loss += loss.item()
                    val_batches += 1
                    if logits.ndim == 2 and logits.shape[1] >= 2:
                        probs = torch.softmax(logits, dim=-1)[:, 1]
                    else:
                        probs = torch.sigmoid(logits.view(-1))
                    val_preds.append(probs.detach())
                    val_labels_list.append(labels.detach())
            
            avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('nan')
            if dist.is_available() and dist.is_initialized() and world_size > 1:
                stats_tensor = torch.tensor([total_val_loss, float(val_batches)], device=device, dtype=torch.float64)
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                global_loss = stats_tensor[0].item()
                global_batches = stats_tensor[1].item()
                avg_val_loss = global_loss / global_batches if global_batches > 0 else float('nan')
            val_phase_time = time.time() - val_phase_start
            
            if val_preds:
                val_pred_tensor = torch.cat(val_preds)
                val_label_tensor = torch.cat(val_labels_list).to(torch.long)
            else:
                val_pred_tensor = torch.empty(0, device=device, dtype=torch.float32)
                val_label_tensor = torch.empty(0, device=device, dtype=torch.long)

            gathered_val_preds = gather_variable_tensor(val_pred_tensor, world_size, device)
            gathered_val_labels = gather_variable_tensor(val_label_tensor, world_size, device)

            if rank == 0:
                val_scores = torch.cat(gathered_val_preds).cpu().to(torch.float32).numpy() if gathered_val_preds else np.array([])
                val_labels_all = torch.cat(gathered_val_labels).cpu().to(torch.float32).numpy() if gathered_val_labels else np.array([])
                if len(np.unique(val_labels_all)) >= 2:
                    val_auprc = metrics.average_precision_score(val_labels_all, val_scores)
                    val_auroc = metrics.roc_auc_score(val_labels_all, val_scores)
                    val_preds_binary = (val_scores > 0.5).astype(int)
                    val_precision = metrics.precision_score(val_labels_all, val_preds_binary)
                    val_recall = metrics.recall_score(val_labels_all, val_preds_binary)
                    val_f1 = metrics.f1_score(val_labels_all, val_preds_binary)
                    val_acc = metrics.accuracy_score(val_labels_all, val_preds_binary)
                    val_mcc = metrics.matthews_corrcoef(val_labels_all, val_preds_binary)
                else:
                    val_auprc = float('nan')
                    val_auroc = float('nan')
                    val_precision = float('nan')
                    val_recall = float('nan')
                    val_f1 = float('nan')
                    val_acc = float('nan')
                    val_mcc = float('nan')

                current_train_metrics = {
                    'epoch': epoch + 1,
                    'loss': float(avg_train_loss),
                    'auprc': float(train_auprc) if isinstance(train_auprc, (float, np.floating)) else float(train_auprc),
                    'auroc': float(train_auroc) if isinstance(train_auroc, (float, np.floating)) else float(train_auroc),
                    'precision': float(train_precision) if isinstance(train_precision, (float, np.floating)) else float(train_precision),
                    'recall': float(train_recall) if isinstance(train_recall, (float, np.floating)) else float(train_recall),
                    'f1': float(train_f1) if isinstance(train_f1, (float, np.floating)) else float(train_f1),
                    'accuracy': float(train_acc) if isinstance(train_acc, (float, np.floating)) else float(train_acc),
                    'mcc': float(train_mcc) if isinstance(train_mcc, (float, np.floating)) else float(train_mcc),
                }
                current_val_metrics = {
                    'epoch': epoch + 1,
                    'loss': float(avg_val_loss),
                    'auprc': float(val_auprc) if isinstance(val_auprc, (float, np.floating)) else float(val_auprc),
                    'auroc': float(val_auroc) if isinstance(val_auroc, (float, np.floating)) else float(val_auroc),
                    'precision': float(val_precision) if isinstance(val_precision, (float, np.floating)) else float(val_precision),
                    'recall': float(val_recall) if isinstance(val_recall, (float, np.floating)) else float(val_recall),
                    'f1': float(val_f1) if isinstance(val_f1, (float, np.floating)) else float(val_f1),
                    'accuracy': float(val_acc) if isinstance(val_acc, (float, np.floating)) else float(val_acc),
                    'mcc': float(val_mcc) if isinstance(val_mcc, (float, np.floating)) else float(val_mcc),
                }
                val_components = {
                    'auroc': current_val_metrics['auroc'],
                    'auprc': current_val_metrics['auprc'],
                    'f1': current_val_metrics['f1'],
                    'accuracy': current_val_metrics['accuracy'],
                    'mcc': current_val_metrics['mcc'],
                }
                if all(np.isfinite(list(val_components.values()))):
                    current_combined_score = sum(
                        val_components[key] * COMBINED_METRIC_WEIGHTS[key]
                        for key in COMBINED_METRIC_WEIGHTS
                    )
                else:
                    current_combined_score = float('nan')
            else:
                val_auprc = val_auroc = val_precision = val_recall = val_f1 = val_acc = val_mcc = 0.0
            
            epoch_time = time.time() - epoch_start_time
            cumulative_time = time.time() - start_time
            
            if rank == 0 and val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch_auprc = epoch + 1
                try:
                    if not check_disk_space(str(best_auprc_model.parent), 1.0):
                        print("⚠️ Insufficient disk space, skipping model save")
                    else:
                        best_auprc_model.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.module.state_dict(), best_auprc_model)
                        best_val_metrics = dict(current_val_metrics)
                        best_train_metrics = dict(current_train_metrics)
                        print(f"✅ Saved best AUPRC model: {best_auprc:.4f}")
                except Exception as e:
                    print(f"⚠️ Failed to save best AUPRC model: {e}")
                    pass
            
            if rank == 0 and val_auroc > best_auroc:
                best_auroc = val_auroc
                best_epoch_auroc = epoch + 1
                try:
                    if not check_disk_space(str(best_auroc_model.parent), 1.0):
                        print("⚠️ Insufficient disk space, skipping model save")
                    else:
                        best_auroc_model.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.module.state_dict(), best_auroc_model)
                        print(f"✅ Saved best AUROC model: {best_auroc:.4f}")
                except Exception as e:
                    print(f"⚠️ Failed to save best AUROC model: {e}")
                    pass
            
            if rank == 0:
                print("Training metrics:")
                print(f"  Loss: {avg_train_loss:.4f}")
                print(f"  AUPRC: {train_auprc:.4f}, AUROC: {train_auroc:.4f}")
                print(f"  F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                print(f"  Accuracy: {train_acc:.4f}, MCC: {train_mcc:.4f}")
                print("Validation metrics:")
                print(f"  Loss: {avg_val_loss:.4f}")
                print(f"  AUPRC: {val_auprc:.4f}, AUROC: {val_auroc:.4f}")
                print(f"  F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
                print(f"  Accuracy: {val_acc:.4f}, MCC: {val_mcc:.4f}")
                print(f"Best - AUPRC: {best_auprc:.4f}@{best_epoch_auprc}, AUROC: {best_auroc:.4f}@{best_epoch_auroc}")
                print(f"Elapsed: {epoch_time:.2f}s (train: {train_phase_time:.1f}s, val: {val_phase_time:.1f}s)")
                print(f"Memory: {get_memory_usage()}")
                if np.isfinite(current_combined_score):
                    print("Combined metric (AUROC*35% + AUPRC*30% + F1*15% + Acc*5% + MCC*15%): "
                          f"{current_combined_score:.4f}")
                else:
                    print("Combined metric: cannot be computed (insufficient validation classes or invalid values)")

                if np.isfinite(current_combined_score):
                    if not np.isfinite(best_combined_score) or current_combined_score > best_combined_score:
                        best_combined_score = current_combined_score
                        best_combined_epoch = epoch + 1
                        no_improve_epochs = 0
                        print(f"Combined metric improved, current best {current_combined_score:.4f} (epoch {epoch+1})")
                    else:
                        no_improve_epochs += 1
                else:
                    no_improve_epochs += 1

                if epoch + 1 < EARLY_STOP_MIN_EPOCHS:
                    early_stop_status = f"MinEpochs({epoch+1}/{EARLY_STOP_MIN_EPOCHS})"
                else:
                    early_stop_status = f"NoImprove({no_improve_epochs}/{EARLY_STOP_PATIENCE})"

                if (epoch + 1) >= EARLY_STOP_MIN_EPOCHS and no_improve_epochs >= EARLY_STOP_PATIENCE:
                    early_stop_triggered = True
                    early_stop_epoch = epoch + 1
                    early_stop_status = f"EarlyStopped(patience={EARLY_STOP_PATIENCE})"
                    print(f"🛑 Early stopping triggered: combined metric did not improve for {EARLY_STOP_PATIENCE} consecutive epochs, stopping at epoch {epoch+1}")
                    if np.isfinite(best_combined_score):
                        print(f"   Best combined metric: {best_combined_score:.4f} (epoch {best_combined_epoch})")

                if not early_stop_triggered and avg_train_loss < EARLY_STOP_TRAIN_LOSS_THRESHOLD and (epoch + 1) >= EARLY_STOP_MIN_EPOCHS:
                    early_stop_triggered = True
                    early_stop_epoch = epoch + 1
                    early_stop_status = f"EarlyStopped(train_loss<{EARLY_STOP_TRAIN_LOSS_THRESHOLD})"
                    print(f"🛑 Early stopping triggered: training loss reached threshold {EARLY_STOP_TRAIN_LOSS_THRESHOLD}, stopping at epoch {epoch+1}")

                if not early_stop_triggered:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        val_loss_no_improve_epochs = 0
                    else:
                        val_loss_no_improve_epochs += 1
                        if val_loss_no_improve_epochs >= EARLY_STOP_VAL_LOSS_PATIENCE and (epoch + 1) >= EARLY_STOP_MIN_EPOCHS:
                            early_stop_triggered = True
                            early_stop_epoch = epoch + 1
                            early_stop_status = f"EarlyStopped(val_loss_no_improve={EARLY_STOP_VAL_LOSS_PATIENCE})"
                            print(f"🛑 Early stopping triggered: validation loss did not improve for {EARLY_STOP_VAL_LOSS_PATIENCE} epochs, stopping at epoch {epoch+1}")
                            print(f"   Best validation loss: {best_val_loss:.4f}")

                print(f"Early stop status: {early_stop_status}")
                
                with open(epoch_timing_file, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch+1}\t{epoch_time:.2f}\t{epoch_time/60:.2f}\t{train_phase_time:.2f}\t{val_phase_time:.2f}\t{cumulative_time:.2f}\n")
                
                with open(timing_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Epoch {epoch+1}/{EPOCHS}: {epoch_time:.2f}s (cumulative: {cumulative_time/60:.1f}min)\n")
                    if (epoch + 1) % 10 == 0:
                        avg_epoch_time = cumulative_time / (epoch + 1)
                        remaining_epochs = EPOCHS - (epoch + 1)
                        estimated_remaining = remaining_epochs * avg_epoch_time
                        f.write(f"  Average per epoch: {avg_epoch_time:.2f}s, Estimated remaining: {estimated_remaining/60:.1f}min\n")
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.2f}s)\n")
                    f.write(
                        "Train - "
                        f"Loss: {avg_train_loss:.4f}, "
                        f"AUPRC: {train_auprc:.4f}, AUROC: {train_auroc:.4f}, "
                        f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, "
                        f"F1: {train_f1:.4f}, Accuracy: {train_acc:.4f}, MCC: {train_mcc:.4f}\n"
                    )
                    f.write(
                        "Val   - "
                        f"Loss: {avg_val_loss:.4f}, "
                        f"AUPRC: {val_auprc:.4f}, AUROC: {val_auroc:.4f}, "
                        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                        f"F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}, MCC: {val_mcc:.4f}\n"
                    )
                    f.write(f"Best  - AUPRC: {best_auprc:.4f}@{best_epoch_auprc}, AUROC: {best_auroc:.4f}@{best_epoch_auroc}\n")
                    combo_display = f"{current_combined_score:.4f}" if np.isfinite(current_combined_score) else "nan"
                    best_combo_display = f"{best_combined_score:.4f}" if np.isfinite(best_combined_score) else "nan"
                    f.write(
                        f"Combo  - Score: {combo_display}, Status: {early_stop_status}, "
                        f"Patience: {no_improve_epochs}/{EARLY_STOP_PATIENCE}, "
                        f"Best: {best_combo_display}@{best_combined_epoch if best_combined_epoch else '-'}\n"
                    )

            if dist.is_available() and dist.is_initialized() and world_size > 1:
                stop_tensor = torch.tensor([1 if early_stop_triggered else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                early_stop_triggered = bool(stop_tensor.item())

            if early_stop_triggered:
                break
        
    total_training_time = time.time() - start_time

    if rank == 0:
        print()
        print(f"{'='*60}")
        print("📊 Starting evaluation...")
        print(f"{'='*60}")
        print("\n[Test 1] Using best AUPRC model...")
        try:
            if best_auprc_model.exists():
                model.module.load_state_dict(torch.load(best_auprc_model))
                print(f"✅ Loaded best AUPRC model: {best_auprc_model}")
            else:
                print(f"⚠️ Best AUPRC model file does not exist: {best_auprc_model}, using current model")
        except Exception as e:
            print(f"⚠️ Failed to load best AUPRC model: {e}, using current model")
        
        model.eval()
        test_preds_auprc = []
        test_labels_auprc = []

        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue
                g1, g2, labels = batch
                g1, g2 = g1.to(device), g2.to(device)
                labels = labels.long().to(device)

                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if device.type == 'cuda' else torch.cuda.amp.autocast(enabled=False):
                    logits = model(g1, g2)
                    logits = torch.clamp(logits, min=-20.0, max=20.0)

                if logits.ndim == 2 and logits.shape[1] >= 2:
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                else:
                    probs = torch.sigmoid(logits.view(-1))
                test_preds_auprc.extend(probs.cpu().to(torch.float32).numpy())
                test_labels_auprc.extend(labels.cpu().to(torch.float32).numpy())

        test_preds_arr_auprc = np.array(test_preds_auprc)
        if test_preds_arr_auprc.ndim == 2 and test_preds_arr_auprc.shape[1] == 2:
            test_scores_auprc = test_preds_arr_auprc[:, 1]
        else:
            test_scores_auprc = test_preds_arr_auprc
        
        if len(np.unique(test_labels_auprc)) >= 2:
            test_auprc_model_auprc = metrics.average_precision_score(test_labels_auprc, test_scores_auprc)
            test_auprc_model_auroc = metrics.roc_auc_score(test_labels_auprc, test_scores_auprc)
            test_preds_binary_auprc = (test_scores_auprc > 0.5).astype(int)
            test_auprc_model_f1 = metrics.f1_score(test_labels_auprc, test_preds_binary_auprc)
            test_auprc_model_acc = metrics.accuracy_score(test_labels_auprc, test_preds_binary_auprc)
            test_auprc_model_mcc = metrics.matthews_corrcoef(test_labels_auprc, test_preds_binary_auprc)
            test_auprc_model_precision = metrics.precision_score(test_labels_auprc, test_preds_binary_auprc)
            test_auprc_model_recall = metrics.recall_score(test_labels_auprc, test_preds_binary_auprc)
        else:
            test_auprc_model_auprc = float('nan')
            test_auprc_model_auroc = float('nan')
            test_auprc_model_f1 = float('nan')
            test_auprc_model_acc = float('nan')
            test_auprc_model_mcc = float('nan')
            test_auprc_model_precision = float('nan')
            test_auprc_model_recall = float('nan')

        print("\nTest metrics (based on best AUPRC model):")
        print(f"  AUPRC: {test_auprc_model_auprc:.4f}, AUROC: {test_auprc_model_auroc:.4f}")
        print(f"  F1: {test_auprc_model_f1:.4f}, Precision: {test_auprc_model_precision:.4f}, Recall: {test_auprc_model_recall:.4f}")
        print(f"  Accuracy: {test_auprc_model_acc:.4f}, MCC: {test_auprc_model_mcc:.4f}")

        try:
            with open(test_metrics_auprc_file, 'w', encoding='utf-8') as f:
                f.write(f"Ablation experiment test results: {exp_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Experiment description: {experiment_config.get('description', 'N/A')}\n")
                f.write(f"Scientific hypothesis: {experiment_config.get('scientific_hypothesis', 'N/A')}\n")
                f.write(f"\nModel type: Best AUPRC model\n")
                f.write(f"Model file: {best_auprc_model}\n")
                f.write(f"\n{'='*60}\n")
                f.write(f"Test metrics:\n")
                f.write(f"  AUPRC: {test_auprc_model_auprc:.4f}\n")
                f.write(f"  AUROC: {test_auprc_model_auroc:.4f}\n")
                f.write(f"  F1 Score: {test_auprc_model_f1:.4f}\n")
                f.write(f"  Precision: {test_auprc_model_precision:.4f}\n")
                f.write(f"  Recall: {test_auprc_model_recall:.4f}\n")
                f.write(f"  Accuracy: {test_auprc_model_acc:.4f}\n")
                f.write(f"  MCC: {test_auprc_model_mcc:.4f}\n")
                f.write(f"\n{'='*60}\n")
                f.write(f"Training information:\n")
                f.write(f"  Total training time: {total_training_time/60:.2f} minutes\n")
                f.write(f"  Best validation AUPRC: {best_auprc:.4f} (epoch {best_epoch_auprc})\n")
                f.write(f"  Best validation AUROC: {best_auroc:.4f} (epoch {best_epoch_auroc})\n")
                f.write(f"\nConfiguration parameters:\n")
                for key, value in experiment_config.items():
                    if key not in ['description', 'scientific_hypothesis']:
                        f.write(f"  {key}: {value}\n")
            print(f"✅ AUPRC model test results saved to: {test_metrics_auprc_file}")
        except Exception as e:
            print(f"⚠️ Failed to save AUPRC model test results: {e}")

        print("\n[Test 2] Using best AUROC model...")
        try:
            if best_auroc_model.exists():
                model.module.load_state_dict(torch.load(best_auroc_model))
                print(f"✅ Loaded best AUROC model: {best_auroc_model}")
            else:
                print(f"⚠️ Best AUROC model file does not exist: {best_auroc_model}, skipping AUROC model evaluation")
                test_auroc_model_auprc = float('nan')
                test_auroc_model_auroc = float('nan')
                test_auroc_model_f1 = float('nan')
                test_auroc_model_acc = float('nan')
                test_auroc_model_mcc = float('nan')
                test_auroc_model_precision = float('nan')
                test_auroc_model_recall = float('nan')
        except Exception as e:
            print(f"⚠️ Failed to load best AUROC model: {e}, skipping AUROC model evaluation")
            test_auroc_model_auprc = float('nan')
            test_auroc_model_auroc = float('nan')
            test_auroc_model_f1 = float('nan')
            test_auroc_model_acc = float('nan')
            test_auroc_model_mcc = float('nan')
            test_auroc_model_precision = float('nan')
            test_auroc_model_recall = float('nan')
        else:
            model.eval()
            test_preds_auroc = []
            test_labels_auroc = []

            with torch.no_grad():
                for batch in test_loader:
                    if batch is None:
                        continue
                    g1, g2, labels = batch
                    g1, g2 = g1.to(device), g2.to(device)
                    labels = labels.long().to(device)

                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if device.type == 'cuda' else torch.cuda.amp.autocast(enabled=False):
                        logits = model(g1, g2)
                        logits = torch.clamp(logits, min=-20.0, max=20.0)

                    if logits.ndim == 2 and logits.shape[1] >= 2:
                        probs = torch.softmax(logits, dim=-1)[:, 1]
                    else:
                        probs = torch.sigmoid(logits.view(-1))
                    test_preds_auroc.extend(probs.cpu().to(torch.float32).numpy())
                    test_labels_auroc.extend(labels.cpu().to(torch.float32).numpy())

            test_preds_arr_auroc = np.array(test_preds_auroc)
            if test_preds_arr_auroc.ndim == 2 and test_preds_arr_auroc.shape[1] == 2:
                test_scores_auroc = test_preds_arr_auroc[:, 1]
            else:
                test_scores_auroc = test_preds_arr_auroc
            
            if len(np.unique(test_labels_auroc)) >= 2:
                test_auroc_model_auprc = metrics.average_precision_score(test_labels_auroc, test_scores_auroc)
                test_auroc_model_auroc = metrics.roc_auc_score(test_labels_auroc, test_scores_auroc)
                test_preds_binary_auroc = (test_scores_auroc > 0.5).astype(int)
                test_auroc_model_f1 = metrics.f1_score(test_labels_auroc, test_preds_binary_auroc)
                test_auroc_model_acc = metrics.accuracy_score(test_labels_auroc, test_preds_binary_auroc)
                test_auroc_model_mcc = metrics.matthews_corrcoef(test_labels_auroc, test_preds_binary_auroc)
                test_auroc_model_precision = metrics.precision_score(test_labels_auroc, test_preds_binary_auroc)
                test_auroc_model_recall = metrics.recall_score(test_labels_auroc, test_preds_binary_auroc)
            else:
                test_auroc_model_auprc = float('nan')
                test_auroc_model_auroc = float('nan')
                test_auroc_model_f1 = float('nan')
                test_auroc_model_acc = float('nan')
                test_auroc_model_mcc = float('nan')
                test_auroc_model_precision = float('nan')
                test_auroc_model_recall = float('nan')

            print("\nTest metrics (based on best AUROC model):")
            print(f"  AUPRC: {test_auroc_model_auprc:.4f}, AUROC: {test_auroc_model_auroc:.4f}")
            print(f"  F1: {test_auroc_model_f1:.4f}, Precision: {test_auroc_model_precision:.4f}, Recall: {test_auroc_model_recall:.4f}")
            print(f"  Accuracy: {test_auroc_model_acc:.4f}, MCC: {test_auroc_model_mcc:.4f}")

        try:
            with open(test_metrics_auroc_file, 'w', encoding='utf-8') as f:
                f.write(f"Ablation experiment test results: {exp_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Experiment description: {experiment_config.get('description', 'N/A')}\n")
                f.write(f"Scientific hypothesis: {experiment_config.get('scientific_hypothesis', 'N/A')}\n")
                f.write(f"\nModel type: Best AUROC model\n")
                f.write(f"Model file: {best_auroc_model}\n")
                f.write(f"\n{'='*60}\n")
                f.write(f"Test metrics:\n")
                f.write(f"  AUPRC: {test_auroc_model_auprc:.4f}\n")
                f.write(f"  AUROC: {test_auroc_model_auroc:.4f}\n")
                f.write(f"  F1 Score: {test_auroc_model_f1:.4f}\n")
                f.write(f"  Precision: {test_auroc_model_precision:.4f}\n")
                f.write(f"  Recall: {test_auroc_model_recall:.4f}\n")
                f.write(f"  Accuracy: {test_auroc_model_acc:.4f}\n")
                f.write(f"  MCC: {test_auroc_model_mcc:.4f}\n")
                f.write(f"\n{'='*60}\n")
                f.write(f"Training information:\n")
                f.write(f"  Total training time: {total_training_time/60:.2f} minutes\n")
                f.write(f"  Best validation AUPRC: {best_auprc:.4f} (epoch {best_epoch_auprc})\n")
                f.write(f"  Best validation AUROC: {best_auroc:.4f} (epoch {best_epoch_auroc})\n")
                f.write(f"\nConfiguration parameters:\n")
                for key, value in experiment_config.items():
                    if key not in ['description', 'scientific_hypothesis']:
                        f.write(f"  {key}: {value}\n")
            print(f"✅ AUROC model test results saved to: {test_metrics_auroc_file}")
        except Exception as e:
            print(f"⚠️ Failed to save AUROC model test results: {e}")

        test_auprc = test_auprc_model_auprc
        test_auroc = test_auprc_model_auroc
        test_f1 = test_auprc_model_f1
        test_acc = test_auprc_model_acc
        test_mcc = test_auprc_model_mcc
        test_precision = test_auprc_model_precision
        test_recall = test_auprc_model_recall

        best_val_report = format_metric_dict(best_val_metrics) or format_metric_dict(current_val_metrics)
        best_train_report = format_metric_dict(best_train_metrics) or format_metric_dict(current_train_metrics)
        test_report = {
            'auprc': sanitize_metric(test_auprc),
            'auroc': sanitize_metric(test_auroc),
            'f1': sanitize_metric(test_f1),
            'accuracy': sanitize_metric(test_acc),
            'mcc': sanitize_metric(test_mcc),
            'precision': sanitize_metric(test_precision),
            'recall': sanitize_metric(test_recall)
        }
        early_stopping_report = {
            'enabled': EARLY_STOP_PATIENCE > 0,
            'triggered': bool(early_stop_triggered),
            'epoch': int(early_stop_epoch) if early_stop_epoch is not None else None,
            'status': early_stop_status,
            'patience': EARLY_STOP_PATIENCE,
            'min_epochs': EARLY_STOP_MIN_EPOCHS,
            'best_combined_score': sanitize_metric(best_combined_score),
            'best_combined_epoch': int(best_combined_epoch) if best_combined_epoch is not None else None,
            'final_combined_score': sanitize_metric(current_combined_score),
            'patience_counter': no_improve_epochs,
        }

        results = {
            'experiment': exp_name,
            'description': experiment_config['description'],
            'scientific_hypothesis': experiment_config['scientific_hypothesis'],
            'config_changes': {k: v for k, v in experiment_config.items() if k not in ['description', 'scientific_hypothesis']},
            'total_epochs': EPOCHS,
            'training_time_seconds': total_training_time,
            'best_val_auprc': sanitize_metric(best_auprc),
            'best_val_auroc': sanitize_metric(best_auroc),
            'best_epoch_auprc': best_epoch_auprc,
            'best_epoch_auroc': best_epoch_auroc,
            'best_train_metrics': best_train_report,
            'best_val_metrics': best_val_report,
            'early_stopping': early_stopping_report,
            'combined_metric_weights': COMBINED_METRIC_WEIGHTS,
            'test_results': test_report
        }

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️ Failed to save results file: {e}")

        print()
        print("✅ Experiment complete!")
        print("   Total elapsed time: " + f"{total_training_time/60:.2f}" + " minutes")
        print("   Results saved to: " + str(exp_output_dir))
        print("="*60)
        print()

    clear_memory()
    cleanup()
    
def main():
    """B1_NoLTC standalone training entry point."""
    parser = argparse.ArgumentParser(description='B1_NoLTC standalone training')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--world_size', type=int, default=3, help='Number of GPUs to use')
    parser.add_argument('--train_batch_size', type=int, default=None, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=None, help='Validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=None, help='Test batch size')
    parser.add_argument('--seed', type=int, default=867482, help='Random seed')

    args = parser.parse_args()

    rank = 0

    print("🔧 Parsed arguments:")
    print(f"  epochs: {args.epochs}")
    print(f"  world_size: {args.world_size}")
    print(f"  train_batch_size: {args.train_batch_size}")
    print(f"  val_batch_size: {args.val_batch_size}")
    print(f"  test_batch_size: {args.test_batch_size}")
    print(f"  seed: {args.seed}")

    setup_seed(args.seed)
    if any(v is not None for v in (args.train_batch_size, args.val_batch_size, args.test_batch_size)):
        update_batch_sizes(
            train=args.train_batch_size,
            val=args.val_batch_size,
            test=args.test_batch_size,
        )

    ensure_output_dirs()

    print("Loading data splits...")
    train_samples, val_samples, test_samples = load_predefined_splits()
    all_samples = train_samples + val_samples + test_samples
    print(f"Building samples batches for {len(all_samples)} samples...")
    build_samples_from_list(all_samples, SAMPLES_BATCH_DIR, rank)
    ds_all = LazySampleDataset(SAMPLES_BATCH_DIR)
    train_idx = list(range(len(train_samples)))
    val_idx = list(range(len(train_samples), len(train_samples) + len(val_samples)))
    test_idx = list(range(len(train_samples) + len(val_samples), len(all_samples)))
    output_dir = OUTPUT_ROOT
    world_size = args.world_size
    port = find_free_port(BASE_PORT)

    print(f"Starting distributed training with {world_size} GPUs...")
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Test samples: {len(test_samples)}")

    try:
        mp.spawn(
            train,
            args=(world_size, port, train_idx, val_idx, test_idx, BASE_CONFIG, 'B1_NoLTC_dc'),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()