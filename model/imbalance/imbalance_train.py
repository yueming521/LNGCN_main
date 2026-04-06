import torch
import torch.nn as nn
import sklearn.metrics as metrics
from tqdm import tqdm
from torch.utils.data import DistributedSampler, Dataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, recall_score, precision_score, auc, \
    roc_auc_score, matthews_corrcoef, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP, FullyConnected, Wiring
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.utils.data as data
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import sys
import traceback
import time

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import data_use_imbalance
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import glob
import pickle
import numpy as np
import gc
import socket

from torch.utils.data import DataLoader
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP, FullyConnected, Wiring
import dgl
import dgl
from dgl.nn import GraphConv, Set2Set

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
current_dir = os.path.dirname(os.path.abspath(__file__))
DEBUG_SHAPES = os.getenv("PPI_DEBUG_SHAPES", "0") == "1"

from pathlib import Path as _Path
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = _Path(_THIS_FILE).parent.parent 
OUTPUT_ROOT = _Path(os.getenv("PPI_OUTPUT_DIR", str(_PROJECT_ROOT / "LNGCN_main/results/imbalance_human")))

def ensure_output_dirs():
    try:
        (OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "fold_data_splits").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def p(*parts) -> str:
    return str(OUTPUT_ROOT.joinpath(*parts))

def paths_for_fold(fold: int):
    return {
        'log': p(f"imbalance_human-enhanced-earlystop_fold{fold}_training_log.txt"),
        'nan': p(f"imbalance_human-enhanced-earlystop_fold{fold}_nan_statistics.txt"),
        'best_auprc': p(f"imbalance_human-enhanced-earlystop_fold{fold}_best_auprc.pt"),
        'best_auroc': p(f"imbalance_human-enhanced-earlystop_fold{fold}_best_auroc.pt"),
        'best_f1': p(f"imbalance_human-enhanced-earlystop_fold{fold}_best_f1.pt"),
        'test_results_auprc': p(f"fold_{fold}_test_results_best_auprc.txt"),
        'test_results_auroc': p(f"fold_{fold}_test_results_best_auroc.txt"),
        'test_results_f1': p(f"fold_{fold}_test_results_best_f1.txt"),
        'fold_result': p(f"fold_{fold}_results.json"),
        'timing': p(f"fold_{fold}_timing_log.txt"),
        'status': p(f"imbalance_human-status_fold{fold}.txt"),
        'error': p(f"imbalance_human-error_fold{fold}.txt"),
        'error_detailed': p(f"imbalance_human-error_fold{fold}_detailed.txt"),
        'error_main': p(f"imbalance_human-error_fold{fold}_main.txt"),
    }

def common_paths():
    return {
        'kfold_summary': p("imbalance_human-ldmk-kfold_results_summary.txt"),
        'final_results': p("imbalance_human-ldmk-kfold_final_results.txt"),
        'fold_data_splits_dir': p("fold_data_splits"),
        'samples_batches': p("imbalance_human_all-samples_batches"),
    }


def find_free_port(preferred: int | None = None, max_tries: int = 32) -> int:
    def _is_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                sock.listen(1)
                return True
            except OSError:
                return False

    if preferred is not None:
        for offset in range(max_tries):
            candidate = preferred + offset
            if _is_free(candidate):
                return candidate

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        sock.listen(1)
        return sock.getsockname()[1]

STRICT_TIMESTEP = os.getenv("PPI_STRICT_TIMESTEP", "1") == "1"
DDP_DEBUG = os.getenv("PPI_DDP_DEBUG", "0") == "1"

try:
    SILENCE_FUTURE_WARNINGS = os.getenv("PPI_SILENCE_FUTUREWARNINGS", "1") == "1"
    if SILENCE_FUTURE_WARNINGS:
        import warnings
        warnings.simplefilter("ignore", category=FutureWarning)
except Exception:
    pass

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False 
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

EPOCHS = 50
LR = 0.0001
OUTFILE = "ld"

epochs = EPOCHS
lr = LR
nhid = 256
nhidh = 128
nhidhh = 64
dropout = 0.5
time_steps = 5
ode_unfolds = 2
K_FOLDS = 5
BATCH_SIZE = 80 
TRAIN_BATCH_SIZE = int(os.getenv("PPI_TRAIN_BATCH_SIZE", str(BATCH_SIZE)))
VAL_BATCH_SIZE = int(os.getenv("PPI_VAL_BATCH_SIZE", "180"))
TEST_BATCH_SIZE = int(os.getenv("PPI_TEST_BATCH_SIZE", "256"))
SAMPLES_BATCH_DIR = common_paths()['samples_batches']
MAX_WORKERS = min(8, os.cpu_count()) 
BASE_PORT = int(os.getenv("PPI_BASE_PORT", "27519"))

DATALOADER_WORKERS = int(os.getenv(
    "PPI_DATALOADER_WORKERS",
    "4" 
))
PREFETCH_FACTOR = int(os.getenv("PPI_PREFETCH_FACTOR", "2")) 
PIN_MEMORY = os.getenv("PPI_PIN_MEMORY", "1") == "1"
PERSISTENT_WORKERS = os.getenv("PPI_PERSISTENT_WORKERS", "0") == "1"

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

def safe_cleanup_dataloaders(*dataloaders):
    for dl in dataloaders:
        if dl is not None:
            try:
                if hasattr(dl.dataset, 'clear_cache'):
                    dl.dataset.clear_cache()
                shutdown_hooks = []
                iterator = getattr(dl, '_iterator', None)
                if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                    shutdown_hooks.append(iterator._shutdown_workers)
                if hasattr(dl, '_shutdown_workers'):
                    shutdown_hooks.append(dl._shutdown_workers)

                for hook in shutdown_hooks:
                    try:
                        hook()
                    except RuntimeError as hook_err:
                        msg = str(hook_err)
                        if "DataLoader worker" in msg and "killed by signal" in msg:
                            continue
                        raise

                if iterator is not None:
                    dl._iterator = None
            except Exception:
                pass
    clear_memory()

def get_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    return "CUDA not available"


def gather_variable_tensor(tensor: torch.Tensor, world_size: int, device: torch.device):
    if not dist.is_available() or not dist.is_initialized() or world_size == 1:
        return [tensor]
    if tensor is None:
        tensor = torch.empty(0, device=device, dtype=torch.float32)

    tensor = tensor.contiguous()
    leading_dim = tensor.shape[0] if tensor.ndim > 0 else 1
    length_tensor = torch.tensor([leading_dim], device=device, dtype=torch.long)
    gathered_lengths = [torch.zeros_like(length_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, length_tensor)

    max_length = int(max(int(l.item()) for l in gathered_lengths))
    if tensor.ndim == 0:
        tensor = tensor.view(1)
    pad_shape = (max_length - tensor.shape[0],) + tensor.shape[1:]
    if pad_shape[0] > 0:
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
        self.batch_files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.pkl")))
        self.batch_sizes = []
        for fpath in self.batch_files:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            self.batch_sizes.append(len(data))
            del data  
        self.cum_sizes = np.cumsum([0] + self.batch_sizes)
        if indices is not None:
            self.mapping = []
            for idx in indices:
                file_idx = np.searchsorted(self.cum_sizes, idx, 'right') - 1
                pos = idx - self.cum_sizes[file_idx]
                self.mapping.append((file_idx, pos))
        else:
            self.mapping = None

        self._file_cache = {}
        self._cache_size_limit = 3  

    def __len__(self):
        return len(self.mapping) if self.mapping is not None else self.cum_sizes[-1]

    def __getitem__(self, idx):
        file_idx, pos = self.mapping[idx] if self.mapping else (
            np.searchsorted(self.cum_sizes, idx, 'right') - 1,
            idx - self.cum_sizes[np.searchsorted(self.cum_sizes, idx, 'right') - 1]
        )

        if file_idx in self._file_cache:
            batch = self._file_cache[file_idx]
        else:
            with open(self.batch_files[file_idx], "rb") as f:
                batch = pickle.load(f)

            if len(self._file_cache) >= self._cache_size_limit:
                oldest_key = next(iter(self._file_cache))
                del self._file_cache[oldest_key]

            self._file_cache[file_idx] = batch

        return batch[pos]

    def clear_cache(self):
        self._file_cache.clear()
        gc.collect()

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
            timestep_feat = feat[:, -1:]  
        else:
            base_feat = feat
            timestep_feat = torch.zeros(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
        
        h = torch.zeros(base_feat.size(0), self.hidden_dim, device=feat.device, dtype=feat.dtype)
        x_in = base_feat
        for i, cell in enumerate(self.cells):
            t = (i + 1) / float(self.n_layers)
            h = cell(h, x_in, t)
            x_in = h
        
        output = torch.cat([h, timestep_feat], dim=1)  
        return output

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
        distance_times = features[:, -1:]  # [N, 1]
        other_features = features[:, :-1]   # [N, D-1]
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
            x = features[:, :-1]  # [N, nhid]
            distance_times = features[:, -1:]  # [N, 1]
        else:
            x = features if features.shape[1] == self.nhid else features[:, :self.nhid]
            distance_times = torch.zeros(features.size(0), 1, device=features.device, dtype=features.dtype)
        if DEBUG_SHAPES:
            if not hasattr(self, "_shape_logged"):
                print(f"[EnhancedDistanceLTC] in={tuple(features.shape)}, has_time={features.shape[1]==self.nhid+1}, x={tuple(x.shape)}, tcol={tuple(distance_times.shape)}")
                self._shape_logged = True
        
        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            gate = self._compute_enhanced_gating(graph, x, distance_times)
            f = self.transform(gate)
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
        self.scale_fusion = nn.Linear(nhid * 3, nhid)  # LTC + 1-hop + 2-hop
        
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

        scale1_feat = self.multi_scale_conv[0](graph, ltc_feat)  # 1-hop 
        scale2_feat = self.multi_scale_conv[1](graph, scale1_feat)  # 2-hop 

        multi_scale = torch.cat([ltc_feat, scale1_feat, scale2_feat], dim=-1)
        fused_feat = self.scale_fusion(multi_scale)

        x = self.bn(graph_feat + fused_feat)
        x = self.relu(x)

        if self.residual:
            x = x + residual
            x = self.relu(x)

        return x

class LTC(nn.Module):
    def __init__(self, nhid, ode_unfolds=ode_unfolds):
        super().__init__()
        self.nhid = nhid
        self.ode_unfolds = ode_unfolds
        self.tau = nn.Parameter(torch.empty(nhid))
        nn.init.uniform_(self.tau, 0.5, 5)

        self.agg_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        self.time_linear = nn.Linear(nhid + 1, nhid)
        self.transform = nn.Linear(nhid, nhid)
        self.A = nn.Parameter(torch.ones(nhid))
        self.out = nn.Linear(nhid, nhid)

        nn.init.xavier_normal_(self.transform.weight)
        nn.init.xavier_normal_(self.agg_conv.weight)
        nn.init.xavier_normal_(self.time_linear.weight)

    def _compute_hybrid_gating(self, graph, x, t):
        x_agg = self.agg_conv(graph, x)
        t_broadcast = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype) * t
        x_time = torch.cat([x, t_broadcast], dim=1)
        x_time = torch.tanh(self.time_linear(x_time))
        combined = x_agg + x_time
        gate = torch.sigmoid(combined)
        return gate

    def _compute_dynamics(self, graph, x, t):
        gate = self._compute_hybrid_gating(graph, x, t)
        dynamic = self.transform(gate)
        return dynamic

    def _ode_step(self, x, f, delta_t):
        tau = torch.relu(self.tau) + 1e-8
        tau_inv = 1.0 / tau
        numerator = x + delta_t * f * self.A
        denominator = 1 + delta_t * (tau_inv + f)
        denominator = denominator + 1e-12 
        return numerator / denominator

    def forward(self, graph, x, t=0.0, dt=0.01):
        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            f = self._compute_dynamics(graph, x, t)
            x = self._ode_step(x, f, delta_t)
            t += delta_t
        return self.out(x)

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
            t += delta_t
        return self.out(x)

class LTCLayer(nn.Module):
    def __init__(self, nhid, time_steps=time_steps, residual=True):
        super().__init__()
        self.ltc = LTC(nhid=nhid, ode_unfolds=ode_unfolds)
        self.time_steps = time_steps
        self.graph_conv = dgl.nn.GraphConv(nhid, nhid)
        self.bn = nn.LayerNorm(nhid)
        self.relu = nn.ReLU()
        self.residual = residual

    def evolve_features(self, graph, x):
        dt = 1.0 / self.time_steps
        for step in range(self.time_steps):
            t = step * dt
            x = self.ltc(graph, x, t, dt)
        return x

    def forward(self, graph, feat):
        residual = feat
        dynamic_feat = self.evolve_features(graph, feat)
        x = self.graph_conv(graph, dynamic_feat)
        x = self.bn(x)
        x = self.relu(x)
        if self.residual:
            x = x + residual
            x = self.relu(x)
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
        fea1 = checkpoint(self.cfc_preprocess, fea1, use_reentrant=False)
        fea2 = checkpoint(self.cfc_preprocess, fea2, use_reentrant=False)
        if DEBUG_SHAPES and not hasattr(self, "_cfc_logged"):
            print(f"[MyGCN] after CfC fea1={tuple(fea1.shape)}, fea2={tuple(fea2.shape)} (expect (*,{self.nhid+1}))")
            self._cfc_logged = True

        fea1 = self.ltc_conv1(g1, fea1)
        fea1 = self.dropout(fea1)

        fea2 = self.ltc_conv1(g2, fea2)
        fea2 = self.dropout(fea2)
        if DEBUG_SHAPES and not hasattr(self, "_ltc1_logged"):
            print(f"[MyGCN] after LTC1 fea1={tuple(fea1.shape)}, fea2={tuple(fea2.shape)} (expect (*,{self.nhid}))")
            self._ltc1_logged = True

        fea1 = self.ltc_conv2(g1, fea1)
        fea2 = self.ltc_conv2(g2, fea2)
        if DEBUG_SHAPES and not hasattr(self, "_ltc2_logged"):
            print(f"[MyGCN] after LTC2 fea1={tuple(fea1.shape)}, fea2={tuple(fea2.shape)} (expect (*,{self.nhid}))")
            self._ltc2_logged = True
        
        enhanced_fea1 = self.structure_enhance[0](g1, fea1)  # GraphConv
        enhanced_fea1 = self.structure_enhance[1](enhanced_fea1)  # LayerNorm
        enhanced_fea1 = self.structure_enhance[2](enhanced_fea1)  # ReLU
        enhanced_fea1 = self.structure_enhance[3](enhanced_fea1)  # Dropout
        
        enhanced_fea2 = self.structure_enhance[0](g2, fea2)  # GraphConv
        enhanced_fea2 = self.structure_enhance[1](enhanced_fea2)  # LayerNorm
        enhanced_fea2 = self.structure_enhance[2](enhanced_fea2)  # ReLU
        enhanced_fea2 = self.structure_enhance[3](enhanced_fea2)  # Dropout
        
        fea1 = fea1 + enhanced_fea1
        fea2 = fea2 + enhanced_fea2
        
        fea1 = self.dropout(fea1)
        fea2 = self.dropout(fea2)

        g1.ndata['h'] = fea1
        g2.ndata['h'] = fea2

        hg1 = self.pool(g1, g1.ndata['h'])
        hg2 = self.pool(g2, g2.ndata['h'])
        if DEBUG_SHAPES and not hasattr(self, "_pool_logged"):
            print(f"[MyGCN] after pool hg1={tuple(hg1.shape)}, hg2={tuple(hg2.shape)} (expect (*,{2*self.nhid}))")
            self._pool_logged = True

        del g1.ndata['h']
        del g2.ndata['h']

        hg1 = self.projection(hg1)
        hg2 = self.projection(hg2)

        # 1. Element-wise sum (Symmetric)
        hg_sum = hg1 + hg2
        
        # 2. Element-wise product (Symmetric) - capture nonlinear interactions
        hg_prod = hg1 * hg2
        
        # 3. Absolute difference (Symmetric) - capture differences
        hg_diff = torch.abs(hg1 - hg2)
        
        hg = torch.cat([hg_sum, hg_prod, hg_diff], dim=-1)
        
        del hg1, hg2, hg_sum, hg_prod, hg_diff

        h = nn.functional.relu(self.fc1(hg))
        h = self.dropout(h)
        h = nn.functional.relu(self.fc2(h))
        return self.fc3(h)

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

def setup(rank, world_size, port):
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

    import datetime
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600)  
    )
    torch.cuda.set_device(rank)

def cleanup():
    try:
        dist.destroy_process_group()
    except Exception:
        pass

def collate_labels(batch):
    return [item[2] for item in batch]

def build_and_save_samples():
    os.makedirs(SAMPLES_BATCH_DIR, exist_ok=True)
    import glob
    existing_batches = glob.glob(os.path.join(SAMPLES_BATCH_DIR, "batch_*.pkl"))
    if len(existing_batches) > 0:
        print(f"🧹 Cleaning {len(existing_batches)} existing old batch files to ensure data consistency")
        for batch_file in existing_batches:
            try:
                os.remove(batch_file)
            except Exception as e:
                print(f"⚠️ Failed to delete file {batch_file}: {e}")

    data_files = [
        ("jLNGCN_main/data/human_imbalance/29724_neg.txt", 0),
        ("LNGCN_main/data/human_imbalance/3000_pos.txt", 1)
    ]

    print("📋 Collecting all protein IDs...")
    all_protein_ids = set()
    for path, _ in data_files:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        all_protein_ids.update([parts[0], parts[1]])
    
    total_unique = len(all_protein_ids)
    print(f"📊 Found {total_unique} unique proteins, evaluating whether to run full preloading...")

    full_preload_requested = os.getenv("PPI_FULL_COORD_PRELOAD", "1") == "1"
    if full_preload_requested and getattr(data_use_imbalance, "ENABLE_COORD_PRELOAD", True):
        print("🔥 Starting full disk-cache preloading in main process (one-time task)...")
        data_use_imbalance.batch_preload_coordinates(
            list(all_protein_ids), 
            max_workers=min(6, MAX_WORKERS) 
        )
        print("✅ Main-process preloading completed! All folds will share disk cache for much better training efficiency")
        print("🎯 Estimated startup time per fold: 2-5s (disk-cache hits)")
    else:
        skipped_reason = []
        if not full_preload_requested:
            skipped_reason.append("PPI_FULL_COORD_PRELOAD=0")
        if not getattr(data_use_imbalance, "ENABLE_COORD_PRELOAD", True):
            skipped_reason.append("PPI_ENABLE_COORD_PRELOAD=0")
        reason_text = ", ".join(skipped_reason) if skipped_reason else "feature not enabled"
        print(f"⏩ Skipping full coordinate preloading ({reason_text}). Training will lazily load on demand and use LRU/disk cache.")
        print("   To reduce startup memory pressure, set PPI_FULL_COORD_PRELOAD=0 to skip one-time warmup.")

    successful, failed = 0, 0
    batch_idx, current = 0, []
    lock = threading.Lock()

    BATCH_SIZE_SAMPLES = 500

    def worker(p1, p2, lbl):
        try:
            if not data_use_imbalance.check_protein_exists_fast(p1): raise ValueError(f"Protein {p1} not found")
            if not data_use_imbalance.check_protein_exists_fast(p2): raise ValueError(f"Protein {p2} not found")
            return True, (p1, p2, lbl)
        except Exception as e:
            try:
                data_use_imbalance.append_fail_cache(f"{p1},{p2}", f"pair-check: {e}", label="pairs")
            except Exception:
                pass
            return False, None

    for path, label in data_files:
        if not os.path.exists(path):
            continue

        print(f"Processing {path}...")
        pairs_batch = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pairs_batch.append((parts[0], parts[1], label))

                if len(pairs_batch) >= 5000:  
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                        futures = {exe.submit(worker, p1, p2, lbl): (p1, p2) for p1,p2,lbl in pairs_batch}
                        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batch"):
                            ok, sample = fut.result()
                            with lock:
                                if ok:
                                    current.append(sample); successful += 1
                                    if len(current) >= BATCH_SIZE_SAMPLES:
                                        save_path = os.path.join(SAMPLES_BATCH_DIR, f"batch_{batch_idx}.pkl")
                                        with open(save_path, "wb") as f: pickle.dump(current, f)
                                        batch_idx += 1; current = []
                                else:
                                    failed += 1

                    del pairs_batch, futures
                    pairs_batch = []
                    gc.collect()
        if pairs_batch:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                futures = {exe.submit(worker, p1, p2, lbl): (p1, p2) for p1,p2,lbl in pairs_batch}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Final batch"):
                    ok, sample = fut.result()
                    with lock:
                        if ok:
                            current.append(sample); successful += 1
                            if len(current) >= BATCH_SIZE_SAMPLES:
                                save_path = os.path.join(SAMPLES_BATCH_DIR, f"batch_{batch_idx}.pkl")
                                with open(save_path, "wb") as f: pickle.dump(current, f)
                                batch_idx += 1; current = []
                        else:
                            failed += 1
            del pairs_batch, futures

    if current:
        save_path = os.path.join(SAMPLES_BATCH_DIR, f"batch_{batch_idx}.pkl")
        with open(save_path, "wb") as f: pickle.dump(current, f)

    del current
    gc.collect()
    print(f"Samples built: {successful} ok, {failed} failed")

def train(rank, world_size, fold, port, train_idx, val_idx, test_idx):
    try:
        setup(rank, world_size, port)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        if rank == 0:
            print(f"Fold {fold} - Rank {rank}/{world_size}: initializing fold cache (disk-cache based)...")
        
        dataset_meta = data_use_imbalance.load_unified_dataset()
        if rank == 0:
            mode = dataset_meta.get('mode', 'unknown')
            print(f"Fold {fold} - Rank {rank}: data backend = {mode}, total graphs = {dataset_meta.get('count', '?')}")
        graph_cache_size = int(os.getenv("PPI_GRAPH_CACHE_SIZE", "2048"))
        coord_cache_size = int(os.getenv("PPI_COORD_CACHE_SIZE", "20480"))
        data_use_imbalance.init_graph_cache(max_size=graph_cache_size)
        data_use_imbalance.init_coord_cache(max_size=coord_cache_size)
        if rank == 0:
            print(f"Graph cache size: {graph_cache_size}, Coord cache size: {coord_cache_size}")
            if graph_cache_size < 512:
                print("⚠️ Recommended: increase PPI_GRAPH_CACHE_SIZE to 1024+ to reduce repeated graph-build overhead")
        
        DISABLE_WARMUP = os.getenv("PPI_DISABLE_WARMUP", "1") == "1"
        if rank == 0 and not DISABLE_WARMUP:
            print(f"Fold {fold}: starting smart cache warmup (disk-cache based)...")
            
            fold_protein_ids = set()
            ds_temp = LazySampleDataset(SAMPLES_BATCH_DIR, list(train_idx)[:1000]) 
            for i in range(min(500, len(ds_temp))): 
                try:
                    p1, p2, _ = ds_temp[i]
                    fold_protein_ids.update([p1, p2])
                    if len(fold_protein_ids) >= 1000:  
                        break
                except:
                    continue
            
            print(f"Fold {fold}: warming up {len(fold_protein_ids)} core proteins into memory...")
            
            data_use_imbalance.batch_preload_coordinates(
                list(fold_protein_ids), 
                max_workers=2 
            )
            print(f"✅ Fold {fold} cache warmup completed!")
        
        if world_size > 1:
            torch.distributed.barrier()
        
        print(f"Rank {rank}/{world_size} - Fold {fold} initialized!")

        ensure_output_dirs()
        fold_paths = paths_for_fold(fold)
        commons = common_paths()
        log_file = fold_paths['log']
        nan_stats_file = fold_paths['nan']
        timing_file = fold_paths['timing']

        with open(log_file, "w", encoding='utf-8') as f:
            f.write("Epoch\tLoss\tAUPRC\tAUROC\tAcc\tPrecision\tF1\tRecall\tMCC\tEarlyStopping\tCombinedScore\n")

        with open(nan_stats_file, "w", encoding='utf-8') as f:
            f.write("Epoch\tTotal_Batches\tNaN_Batches\tNaN_Percentage\tNaN_Loss\tNaN_Prediction\tNaN_Gradient\n")

        with open(timing_file, "w", encoding='utf-8') as f:
            f.write("Epoch\tTrainBatches\tTrainMean(s)\tTrainP90(s)\tTrainMax(s)\tTrainTotal(s)\tValBatches\tValTotal(s)\n")

        ds_train = LazySampleDataset(SAMPLES_BATCH_DIR, train_idx)
        ds_val = LazySampleDataset(SAMPLES_BATCH_DIR, val_idx)
        ds_test = LazySampleDataset(SAMPLES_BATCH_DIR, test_idx)

        train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True)
        
        cpu_cap = max(1, (os.cpu_count() or 1) - 1)
        requested_workers = max(0, min(DATALOADER_WORKERS, cpu_cap))
        safe_num_workers = max(1, requested_workers) if world_size > 1 else requested_workers
        safe_pin_memory = PIN_MEMORY 
        force_workers_env = os.getenv("PPI_FORCE_WORKERS")
        if force_workers_env is not None:
            try:
                safe_num_workers = max(0, int(force_workers_env))
                if rank == 0:
                    print(f"⚙️ Override DataLoader worker count: {safe_num_workers} (PPI_FORCE_WORKERS)")
            except ValueError:
                if rank == 0:
                    print(f"⚠️ Ignoring invalid PPI_FORCE_WORKERS={force_workers_env}")

        force_pin_env = os.getenv("PPI_FORCE_PIN_MEMORY")
        if force_pin_env is not None:
            if force_pin_env.strip() in {"0", "false", "False"}:
                safe_pin_memory = False
            elif force_pin_env.strip() in {"1", "true", "True"}:
                safe_pin_memory = True
            else:
                if rank == 0:
                    print(f"⚠️ Ignoring invalid PPI_FORCE_PIN_MEMORY={force_pin_env}")
            if rank == 0:
                print(f"⚙️ Override pin_memory: {safe_pin_memory} (PPI_FORCE_PIN_MEMORY)")
        
        effective_prefetch = PREFETCH_FACTOR if (safe_num_workers > 0 and PREFETCH_FACTOR > 0) else None
        persistent_flag = PERSISTENT_WORKERS and safe_num_workers > 0

        if rank == 0:
            print(f"🔧 Multi-GPU safe config: num_workers={safe_num_workers}, pin_memory={safe_pin_memory}, prefetch={effective_prefetch}, "
                  f"train_batch={TRAIN_BATCH_SIZE}")

        loader_common_kwargs = dict(
            pin_memory=safe_pin_memory,
            num_workers=safe_num_workers,
            persistent_workers=persistent_flag
        )
        if safe_num_workers > 0 and effective_prefetch is not None:
            loader_common_kwargs['prefetch_factor'] = effective_prefetch

        train_loader = DataLoader(
            ds_train,
            batch_size=TRAIN_BATCH_SIZE,
            sampler=train_sampler,
            collate_fn=data_use_imbalance.collate_GCN,
            drop_last=True,
            **loader_common_kwargs
        )

        val_sampler = DistributedSampler(ds_val, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if world_size > 1 else None
        val_loader = DataLoader(
            ds_val,
            batch_size=VAL_BATCH_SIZE,
            sampler=val_sampler,
            collate_fn=data_use_imbalance.collate_GCN,
            **loader_common_kwargs
        )

        test_loader = DataLoader(
            ds_test,
            batch_size=TEST_BATCH_SIZE,
            collate_fn=data_use_imbalance.collate_GCN,
            **loader_common_kwargs
        )

        probe_ds = LazySampleDataset(SAMPLES_BATCH_DIR, list(train_idx)[:1])
        probe_loader = DataLoader(
            probe_ds,
            batch_size=1,
            collate_fn=data_use_imbalance.make_collate_GCN(label=f"fold{fold}-probe", every=0),
            num_workers=0,
            pin_memory=PIN_MEMORY
        )
        sample_g1, _, _ = next(iter(probe_loader))
        in_dim = sample_g1.ndata['fea'].shape[1]
        del sample_g1, probe_loader, probe_ds 
        clear_memory()

        if rank == 0:
            print(f"Input dimension: {in_dim}")
            print(f"Before model initialization: {get_memory_usage()}")

        model = MyGCN(in_dim=in_dim, nhid=nhid, dropout=dropout, time_steps=time_steps).to(device)
        model = DDP(
            model,
            device_ids=[rank] if device.type == 'cuda' else None,
            output_device=rank if device.type == 'cuda' else None,
            find_unused_parameters=False,
            broadcast_buffers=True, 
            bucket_cap_mb=25,  
            gradient_as_bucket_view=True  
        )

        if rank == 0:
            print(f"After model initialization: {get_memory_usage()}")

        if rank == 0:
            labels = []

            sample_loader = DataLoader(
                ds_train,
                batch_size=1000,
                collate_fn=collate_labels,
                num_workers=0
            )
            for batch_labels in sample_loader:
                labels.extend(batch_labels)
                if len(labels) > 10000: 
                    break

            classes = np.unique(labels)
            class_weights = compute_class_weight('balanced', classes=classes, y=labels)
            class_weights = torch.tensor(class_weights, device=rank, dtype=torch.float)
        else:
            class_weights = torch.tensor([1.0, 1.0], device=rank, dtype=torch.float)
            
        loss_func = FocalLoss(alpha=class_weights, gamma=2)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler()
        max_grad_norm = 1.0

        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * 0.1)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=total_steps,
            max_lr=lr,
            min_lr=lr / 100
        )

        best_auprc_model_path = fold_paths['best_auprc']
        best_auroc_model_path = fold_paths['best_auroc']
        best_f1_model_path = fold_paths['best_f1']

        fold_metrics = {
            'fold': fold,
            'best_auprc': 0.0,
            'best_auprc_epoch': 0,
            'best_auroc': 0.0,
            'best_auroc_epoch': 0,
            'best_auprc_model_path': best_auprc_model_path,
            'best_auroc_model_path': best_auroc_model_path,
            # Validation set (recorded when best AUPRC is reached)
            'best_val_acc': 0.0,
            'best_val_precision': 0.0,
            'best_val_recall': 0.0,
            'best_val_f1': 0.0,
            'best_val_mcc': 0.0,
            # Test set (detailed metrics for each best model)
            'test_auprc_best_auprc': 0.0,
            'test_auroc_best_auprc': 0.0,
            'test_auprc_best_auroc': 0.0,
            'test_auroc_best_auroc': 0.0,
            'test_acc_best_auprc': 0.0,
            'test_precision_best_auprc': 0.0,
            'test_recall_best_auprc': 0.0,
            'test_f1_best_auprc': 0.0,
            'test_mcc_best_auprc': 0.0,
            'test_acc_best_auroc': 0.0,
            'test_precision_best_auroc': 0.0,
            'test_recall_best_auroc': 0.0,
            'test_f1_best_auroc': 0.0,
            'test_mcc_best_auroc': 0.0,
        }

        best_auprc = 0.0
        best_auprc_epoch = 0
        best_auroc = 0.0
        best_auroc_epoch = 0
        best_f1 = 0.0
        best_f1_epoch = 0

        # Early stopping parameters (tuned for balanced data)
        patience = 15  # Stop if no improvement for 15 consecutive epochs
        min_epochs = 15  # Minimum training epochs
        no_improve_count = 0
        early_stopped = False

        # Early-stopping metric: 5-metric combined strategy (AUROC + AUPRC + F1 + Acc + MCC)
        best_combined_score = 0.0  # Weighted average of 5 metrics

        weights = {
            'auroc': 0.35,      # Most stable, highest weight
            'auprc': 0.30,      # Positive-class quality, core metric
            'f1': 0.15,         # Overall performance, important metric
            'accuracy': 0.05,   # Overall correctness, reference metric
            'mcc': 0.15
        }

        if rank == 0:
            print(f"🚀 Fold {fold} all processes are ready, starting training")

    # Training loop
        for epoch in range(EPOCHS):
            if early_stopped:
                break
            model.train()
            total_loss = 0.0

            train_batches_processed = 0
            train_mean_time = 0.0
            train_p90_time = 0.0
            train_max_time = 0.0
            train_total_time = 0.0
            total_val_batches = 0
            val_duration_global = 0.0

            batch_timing = [] if rank == 0 else None

            try:
                train_sampler.set_epoch(epoch)
            except Exception:
                pass

            if 'val_sampler' in locals() and val_sampler is not None:
                try:
                    val_sampler.set_epoch(epoch)
                except Exception:
                    pass
            if 'test_sampler' in locals() and test_sampler is not None:
                try:
                    test_sampler.set_epoch(epoch)
                except Exception:
                    pass

            total_batches = 0
            nan_loss_count = 0
            nan_prediction_count = 0
            nan_gradient_count = 0
            total_nan_batches = 0

            torch.cuda.empty_cache()

            train_iter = iter(train_loader)
            dl_fallback_done = False  
            num_batches = len(train_loader)
            for batch_idx in range(num_batches):
                total_batches += 1
                batch_has_nan = False
                step_start = time.perf_counter()

                local_ok = 1
                batch = None
                try:
                    if DDP_DEBUG and batch_idx < 3:
                        print(f"[Rank {rank}] waiting for batch {batch_idx}", flush=True)
                    batch = next(train_iter)
                    if DDP_DEBUG and batch_idx < 3:
                        print(f"[Rank {rank}] fetched batch {batch_idx}", flush=True)
                except Exception as e:
                    print(f"⚠️ Rank {rank}: Epoch {epoch} Batch {batch_idx} failed to fetch batch: {e}")
                    if ("DataLoader worker" in str(e) or "worker (pid" in str(e)) and safe_num_workers > 0 and not dl_fallback_done:
                        try:
                            print(f"🔁 Rank {rank}: detected DataLoader worker issue, rebuilding train_loader with num_workers=0 (lower memory/more stable)")
                            train_loader = DataLoader(
                                ds_train,
                                batch_size=TRAIN_BATCH_SIZE,
                                sampler=train_sampler,
                                collate_fn=data_use_imbalance.collate_GCN, 
                                num_workers=0,
                                pin_memory=safe_pin_memory, 
                                drop_last=True,
                                persistent_workers=False
                            )
                            train_iter = iter(train_loader)
                            dl_fallback_done = True
                        except Exception as _rebld_e:
                            print(f"❌ Rank {rank}: train_loader rebuild failed: {_rebld_e}")
                    local_ok = 0
                ok_tensor = torch.tensor([local_ok], device=rank, dtype=torch.int32)
                if dist.is_initialized():
                    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
                if ok_tensor.item() == 0:
                    torch.cuda.empty_cache()
                    continue

                local_ok = 1
                try:
                    batch_g1, batch_g2, batch_label = batch
                    batch_g1 = batch_g1.to(device, non_blocking=PIN_MEMORY)
                    batch_g2 = batch_g2.to(device, non_blocking=PIN_MEMORY)
                    fea1 = batch_g1.ndata['fea'].to(device, dtype=torch.float32, non_blocking=PIN_MEMORY)
                    fea2 = batch_g2.ndata['fea'].to(device, dtype=torch.float32, non_blocking=PIN_MEMORY)
                    batch_label = batch_label.to(device, dtype=torch.long, non_blocking=PIN_MEMORY)
                except Exception as e:
                    print(f"⚠️ Rank {rank}: Epoch {epoch} Batch {batch_idx} data transfer failed: {e}")
                    local_ok = 0
                ok_tensor = torch.tensor([local_ok], device=rank, dtype=torch.int32)
                if dist.is_initialized():
                    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
                if ok_tensor.item() == 0:
                    torch.cuda.empty_cache()
                    continue

                optimizer.zero_grad(set_to_none=True)
                amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if device.type == 'cuda' else torch.cuda.amp.autocast(enabled=False):
                    if epoch == 0 and batch_idx == 0 and rank == 0:
                        try:
                            mdev = next(model.parameters()).device
                        except Exception:
                            mdev = 'unknown'
                        try:
                            gdev = getattr(batch_g1, 'device', 'unknown')
                        except Exception:
                            gdev = 'unknown'
                        print(f"[DeviceCheck] model={mdev}, graph={gdev}, fea1={fea1.device}, cuda_available={torch.cuda.is_available()}")
                    prediction = model(batch_g1, batch_g2, fea1, fea2)
                    prediction = torch.clamp(prediction, min=-20.0, max=20.0)
                    loss = loss_func(prediction, batch_label)

                    nan_detect_local = int(torch.isnan(loss) or torch.isinf(loss) or torch.isnan(prediction).any() or torch.isinf(prediction).any())
                    if nan_detect_local and rank == 0:
                        try:
                            _lv = float(loss.detach().item()) if not (torch.isnan(loss) or torch.isinf(loss)) else float('nan')
                            print(f"⚠️ Epoch {epoch} Batch {batch_idx}: detected NaN/Inf (loss={_lv}); all ranks skip this step")
                        except Exception:
                            print(f"⚠️ Epoch {epoch} Batch {batch_idx}: detected NaN/Inf (loss cannot be converted to scalar); all ranks skip this step")
                        nan_loss_count += 1
                        batch_has_nan = True
                    nan_tensor = torch.tensor([nan_detect_local], device=rank, dtype=torch.int32)
                    if dist.is_initialized():
                        dist.all_reduce(nan_tensor, op=dist.ReduceOp.MAX)
                    if nan_tensor.item() > 0:
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        continue

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    bad_grad_local = int(torch.isnan(grad_norm) or torch.isinf(grad_norm))
                    if bad_grad_local and rank == 0:
                        try:
                            grad_value = float(grad_norm.item())
                            print(f"⚠️ Epoch {epoch} Batch {batch_idx}: NaN/Inf gradient norm = {grad_value}; all ranks skip this step")
                        except Exception:
                            print(f"⚠️ Epoch {epoch} Batch {batch_idx}: NaN/Inf gradient norm (unable to read value); all ranks skip this step")
                        nan_gradient_count += 1
                        batch_has_nan = True
                    bad_grad_tensor = torch.tensor([bad_grad_local], device=rank, dtype=torch.int32)
                    if dist.is_initialized():
                        dist.all_reduce(bad_grad_tensor, op=dist.ReduceOp.MAX)
                    if bad_grad_tensor.item() > 0:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        torch.cuda.empty_cache()
                        continue

                    if rank == 0 and batch_has_nan:
                        total_nan_batches += 1

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    if rank == 0:
                        duration = time.perf_counter() - step_start
                        batch_timing.append((batch_idx, duration))

                except RuntimeError as e:
                    if "NCCL" in str(e) or "collective" in str(e):
                        print(f"🚨 Rank {rank}: NCCL error at batch {batch_idx}: {e}")
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    del batch_g1, batch_g2, fea1, fea2, batch_label, prediction, loss
                    torch.cuda.empty_cache()

            if rank == 0:
                global_total_batches = total_batches * world_size  
                nan_percentage = (total_nan_batches / total_batches * 100) if total_batches > 0 else 0
                print(f"📊 Epoch {epoch} NaN stats (Rank 0): {total_nan_batches}/{total_batches} batches ({nan_percentage:.2f}%) "
                      f"- loss NaN:{nan_loss_count}, prediction NaN:{nan_prediction_count}, gradient NaN:{nan_gradient_count}")

                if batch_timing:
                    durations = np.array([d for _, d in batch_timing], dtype=np.float64)
                    train_batches_processed = len(batch_timing)
                    train_total_time = float(durations.sum())
                    train_mean_time = float(durations.mean())
                    train_p90_time = float(np.percentile(durations, 90))
                    train_max_time = float(durations.max())
                    slowest = sorted(batch_timing, key=lambda x: x[1], reverse=True)[:3]
                    slow_repr = ", ".join(f"idx{idx}:{dur:.2f}s" for idx, dur in slowest)
                    print(f"⏱️ Epoch {epoch} batch timing -> mean {train_mean_time:.2f}s, P90 {train_p90_time:.2f}s, slowest {slow_repr}")
                else:
                    print(f"⏱️ Epoch {epoch} batch timing -> no valid batches collected; all batches may have been skipped or data issues occurred")

                try:
                    with open(nan_stats_file, "a", encoding='utf-8') as f:
                        f.write(f"{epoch}\t{total_batches}\t{total_nan_batches}\t{nan_percentage:.2f}\t"
                               f"{nan_loss_count}\t{nan_prediction_count}\t{nan_gradient_count}\n")
                except Exception as e:
                    print(f"⚠️ Failed to write NaN stats: {e}")
            else:
                nan_percentage = 0
                
            model.eval()
            torch.cuda.empty_cache()

            val_start = time.perf_counter()
            local_val_batches = 0
            local_probs = []
            local_labels = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch is None:
                        continue
                    local_val_batches += 1
                    batch_g1, batch_g2, batch_lbl = batch
                    batch_g1 = batch_g1.to(device, non_blocking=PIN_MEMORY)
                    batch_g2 = batch_g2.to(device, non_blocking=PIN_MEMORY)
                    fea1 = batch_g1.ndata['fea'].to(device, dtype=torch.float32, non_blocking=PIN_MEMORY)
                    fea2 = batch_g2.ndata['fea'].to(device, dtype=torch.float32, non_blocking=PIN_MEMORY)
                    batch_lbl = batch_lbl.to(device, non_blocking=PIN_MEMORY)

                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if device.type == 'cuda' else torch.cuda.amp.autocast(enabled=False):
                        pred = model.module(batch_g1, batch_g2, fea1, fea2)
                        prob = torch.softmax(pred, dim=1)[:, 1]
                        local_probs.append(prob.detach())
                        local_labels.append(batch_lbl.detach())

                    if batch_idx % 5 == 0:
                        del batch_g1, batch_g2, fea1, fea2, batch_lbl, pred, prob
                        torch.cuda.empty_cache()

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            val_duration_local = time.perf_counter() - val_start

            if local_probs:
                local_probs_tensor = torch.cat(local_probs)
                local_labels_tensor = torch.cat(local_labels).to(torch.long)
            else:
                local_probs_tensor = torch.empty(0, device=device, dtype=torch.float32)
                local_labels_tensor = torch.empty(0, device=device, dtype=torch.long)

            val_batches_tensor = torch.tensor([local_val_batches], device=device, dtype=torch.long)
            val_duration_tensor = torch.tensor([val_duration_local], device=device, dtype=torch.float32)

            total_val_batches = local_val_batches
            val_duration_global = val_duration_local
            if dist.is_initialized():
                gathered_batch_counts = [torch.zeros_like(val_batches_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_batch_counts, val_batches_tensor)
                total_val_batches = int(sum(int(t.item()) for t in gathered_batch_counts))

                gathered_val_durations = [torch.zeros_like(val_duration_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_val_durations, val_duration_tensor)
                val_duration_global = max(float(t.item()) for t in gathered_val_durations)

            gathered_probs = gather_variable_tensor(local_probs_tensor, world_size, device)
            gathered_labels = gather_variable_tensor(local_labels_tensor, world_size, device)

            del local_probs, local_labels
            del local_probs_tensor, local_labels_tensor

            if rank == 0:
                val_pred = torch.cat(gathered_probs).cpu().numpy() if gathered_probs else np.array([])
                val_label = torch.cat(gathered_labels).cpu().numpy() if gathered_labels else np.array([])
                if len(np.unique(val_label)) < 2:
                    print(f"Warning: Fold {fold} Epoch {epoch} - Validation set contains only one class!")
                    try:
                        dist.barrier()
                    except Exception:
                        pass
                    model.train()
                    continue

                print(f"⏱️ Fold {fold} Epoch {epoch} validation time -> {val_duration_global:.2f}s, batches {total_val_batches}")
            else:
                val_pred = None
                val_label = None

            del gathered_probs, gathered_labels

            if rank == 0:
                precision, recall, _ = precision_recall_curve(val_label, val_pred)
                auprc = auc(recall, precision)
                auroc = roc_auc_score(val_label, val_pred)
                val_pred_binary = np.array(val_pred) > 0.5
                acc = accuracy_score(val_label, val_pred_binary)
                precision_score_value = precision_score(val_label, val_pred_binary)
                f1 = f1_score(val_label, val_pred_binary)
                recall_score_value = recall_score(val_label, val_pred_binary)
                mcc = matthews_corrcoef(val_label, val_pred_binary)

                combined_score = (auroc * weights['auroc'] +
                                  auprc * weights['auprc'] +
                                  f1 * weights['f1'] +
                                  acc * weights['accuracy'] +
                                  mcc * weights['mcc'])

                improved = False
                if auprc > best_auprc:
                    best_auprc = auprc
                    best_auprc_epoch = epoch
                    try:
                        os.makedirs(os.path.dirname(best_auprc_model_path), exist_ok=True)
                    except Exception:
                        pass
                    try:
                        with open(best_auprc_model_path, 'wb') as f:
                            torch.save(model.module.state_dict(), f)
                    except Exception as e:
                        print(f"❌ Failed to save best AUPRC model: {e}")
                    fold_metrics['best_val_acc'] = acc
                    fold_metrics['best_val_precision'] = precision_score_value
                    fold_metrics['best_val_recall'] = recall_score_value
                    fold_metrics['best_val_f1'] = f1
                    fold_metrics['best_val_mcc'] = mcc
                    print(f"Fold {fold} - Best AUPRC model saved at epoch {epoch} with AUPRC: {auprc:.4f}")

                if auroc > best_auroc:
                    best_auroc = auroc
                    best_auroc_epoch = epoch
                    try:
                        os.makedirs(os.path.dirname(best_auroc_model_path), exist_ok=True)
                    except Exception:
                        pass
                    try:
                        with open(best_auroc_model_path, 'wb') as f:
                            torch.save(model.module.state_dict(), f)
                    except Exception as e:
                        print(f"❌ Failed to save best AUROC model: {e}")
                    print(f"Fold {fold} - Best AUROC model saved at epoch {epoch} with AUROC: {auroc:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_epoch = epoch
                    try:
                        os.makedirs(os.path.dirname(best_f1_model_path), exist_ok=True)
                    except Exception:
                        pass
                    try:
                        with open(best_f1_model_path, 'wb') as f:
                            torch.save(model.module.state_dict(), f)
                    except Exception as e:
                        print(f"❌ Failed to save best F1 model: {e}")
                    print(f"Fold {fold} - Best F1 model saved at epoch {epoch} with F1: {f1:.4f}")

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    no_improve_count = 0
                    improved = True
                    print(f"Fold {fold} - Combined score improved: {combined_score:.4f} (AUROC*35% + AUPRC*30% + F1*15% + Acc*5% + MCC*15%)")
                else:
                    no_improve_count += 1

                early_stop_info = ""
                if epoch >= min_epochs and no_improve_count >= patience:
                    early_stopped = True
                    early_stop_info = f"EarlyStopped(patience={patience})"
                    print(f"🛑 Fold {fold} early stopping triggered: combined score did not improve for {patience} epochs; stopping at epoch {epoch}")
                    print(f"   Best combined score: {best_combined_score:.4f} (AUROC*35% + AUPRC*30% + F1*15% + Acc*5% + MCC*15%)")
                elif epoch < min_epochs:
                    early_stop_info = f"MinEpochs({epoch+1}/{min_epochs})"
                else:
                    early_stop_info = f"NoImprove({no_improve_count}/{patience})"

                avg_loss = total_loss / len(train_loader)
                print(f"Fold {fold} Epoch {epoch}: AUPRC={auprc:.4f}, AUROC={auroc:.4f}, "
                      f"Best AUPRC={best_auprc:.4f}@{best_auprc_epoch}, Best AUROC={best_auroc:.4f}@{best_auroc_epoch}, "
                      f"ACC={acc:.4f}, Precision={precision_score_value:.4f}, F1={f1:.4f}, Recall={recall_score_value:.4f}, MCC={mcc:.4f}, "
                      f"LOSS={avg_loss:.4f}, NaN%={nan_percentage:.2f}%, {early_stop_info}")

                try:
                    log_message = f"{epoch}\t{avg_loss:.4f}\t{auprc:.4f}\t{auroc:.4f}\t{acc:.4f}\t{precision_score_value:.4f}\t{f1:.4f}\t{recall_score_value:.4f}\t{mcc:.4f}\t{early_stop_info}\t{combined_score:.4f}\n"
                    with open(log_file, "a", encoding='utf-8') as f:
                        f.write(log_message)
                except Exception as e:
                    print(f"⚠️ Failed to write log: {e}")

                try:
                    with open(timing_file, "a", encoding='utf-8') as f:
                        f.write(f"{epoch}\t{train_batches_processed}\t{train_mean_time:.4f}\t{train_p90_time:.4f}\t{train_max_time:.4f}\t{train_total_time:.4f}\t{total_val_batches}\t{val_duration_global:.4f}\n")
                except Exception as e:
                    print(f"⚠️ Failed to write timing log: {e}")

            try:
                if dist.is_initialized():
                    dist.barrier()
            except Exception as e:
                print(f"⚠️ Rank {rank}: synchronization after validation failed: {e}")

            model.train()

            if rank == 0:
                print(f"✅ Epoch {epoch} finished (no explicit barrier)")

        try:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                if rank == 0:
                    print(f"🧪 Fold {fold} training completed, all processes synchronized, starting test evaluation...")
        except Exception as e:
            print(f"⚠️ Rank {rank}: synchronization before test failed: {e}")

        # Test set evaluation
        test_auprc_best_auprc = 0.0
        test_auroc_best_auprc = 0.0
        test_auprc_best_auroc = 0.0
        test_auroc_best_auroc = 0.0
        test_auprc_best_f1 = 0.0
        test_auroc_best_f1 = 0.0

        if len(ds_test) > 0 and rank == 0:
            print(f"\n🧪 Starting evaluation of best models for Fold {fold} on the test set...")

            # Thoroughly clean GPU memory before testing
            print("🧹 Cleaning GPU memory from training stage...")
            del train_loader, val_loader 
            del optimizer, scheduler, scaler  
            model.cpu() 
            del model

            # Force garbage collection and GPU memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  
            clear_memory()
            print(f"GPU memory usage after cleanup: {get_memory_usage()}")

            time.sleep(1)

            print("🔄 Rebuilding model for testing...")
            test_model = MyGCN(in_dim=in_dim, nhid=nhid, dropout=dropout, time_steps=time_steps).to(device)
            print(f"GPU memory usage after creating test model: {get_memory_usage()}")

            # Evaluate best AUPRC model
            if os.path.exists(best_auprc_model_path):
                try:
                    print(f"📊 Evaluating best AUPRC model: {best_auprc_model_path}")
                    with open(best_auprc_model_path, 'rb') as f:
                        test_model.load_state_dict(torch.load(f, map_location=f'cuda:{rank}'))
                    test_model.eval()

                    test_pred = []
                    test_label = []

                    with torch.no_grad():
                        for batch_idx, batch in enumerate(test_loader):
                            batch_g1, batch_g2, batch_lbl = batch
                            batch_g1 = batch_g1.to(device)
                            batch_g2 = batch_g2.to(device)
                            fea1 = batch_g1.ndata['fea'].to(device, dtype=torch.float32)
                            fea2 = batch_g2.ndata['fea'].to(device, dtype=torch.float32)
                            batch_lbl = batch_lbl.to(device)

                            outputs = test_model(batch_g1, batch_g2, fea1, fea2)
                            # prob = torch.sigmoid(outputs).squeeze()
                            prob = torch.softmax(outputs, dim=1)[:, 1]

                            test_pred.extend(prob.cpu().numpy())
                            test_label.extend(batch_lbl.cpu().numpy())

                            # Periodically clear memory during testing
                            if batch_idx % 5 == 0:
                                del batch_g1, batch_g2, fea1, fea2, batch_lbl, outputs, prob
                                torch.cuda.empty_cache()

                    if len(np.unique(test_label)) >= 2:
                        # Compute AUPRC and AUROC
                        precision_curve, recall_curve, _ = precision_recall_curve(test_label, test_pred)
                        test_auprc_best_auprc = auc(recall_curve, precision_curve)
                        test_auroc_best_auprc = roc_auc_score(test_label, test_pred)

                        # Compute other evaluation metrics
                        test_pred_binary = np.array(test_pred) > 0.5
                        test_acc = accuracy_score(test_label, test_pred_binary)
                        test_precision = precision_score(test_label, test_pred_binary, zero_division=0)
                        test_recall = recall_score(test_label, test_pred_binary, zero_division=0)
                        test_f1 = f1_score(test_label, test_pred_binary, zero_division=0)
                        test_mcc = matthews_corrcoef(test_label, test_pred_binary)
                        test_cm = confusion_matrix(test_label, test_pred_binary)

                        # Save test metrics for best AUPRC model
                        fold_metrics['test_acc_best_auprc'] = test_acc
                        fold_metrics['test_precision_best_auprc'] = test_precision
                        fold_metrics['test_recall_best_auprc'] = test_recall
                        fold_metrics['test_f1_best_auprc'] = test_f1
                        fold_metrics['test_mcc_best_auprc'] = test_mcc

                        print(f"📈 Best AUPRC model on test set:")
                        print(f"   AUPRC={test_auprc_best_auprc:.4f}, AUROC={test_auroc_best_auprc:.4f}")
                        print(f"   ACC={test_acc:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, MCC={test_mcc:.4f}")
                        print(f"   Confusion Matrix:\n{test_cm}")

                        # Save test results for best AUPRC model
                        test_results_file = fold_paths['test_results_auprc']
                        with open(test_results_file, "w", encoding='utf-8') as f:
                            f.write(f"=== Fold {fold} Test Results - Best AUPRC Model ===\n")
                            f.write(f"Model path: {best_auprc_model_path}\n")
                            f.write(f"Number of test samples: {len(test_label)}\n")
                            f.write(f"Number of positive samples: {sum(test_label)}\n")
                            f.write(f"Number of negative samples: {len(test_label) - sum(test_label)}\n\n")
                            f.write(f"Evaluation metrics:\n")
                            f.write(f"AUPRC: {test_auprc_best_auprc:.6f}\n")
                            f.write(f"AUROC: {test_auroc_best_auprc:.6f}\n")
                            f.write(f"Accuracy: {test_acc:.6f}\n")
                            f.write(f"Precision: {test_precision:.6f}\n")
                            f.write(f"Recall: {test_recall:.6f}\n")
                            f.write(f"F1-Score: {test_f1:.6f}\n")
                            f.write(f"MCC: {test_mcc:.6f}\n")
                            f.write(f"Confusion Matrix:\n{test_cm}\n")
                        print(f"✅ Best AUPRC model test results saved to: {test_results_file}")
                    else:
                        print("⚠️ Test set contains only one class; evaluation metrics cannot be computed")

                except Exception as e:
                    print(f"❌ Error while evaluating best AUPRC model: {e}")

            # Clean memory between model evaluations
            torch.cuda.empty_cache()
            clear_memory()

            # Evaluate best AUROC model
            if os.path.exists(best_auroc_model_path) and best_auroc_model_path != best_auprc_model_path:
                try:
                    print(f"📊 Evaluating best AUROC model: {best_auroc_model_path}")
                    with open(best_auroc_model_path, 'rb') as f:
                        test_model.load_state_dict(torch.load(f, map_location=f'cuda:{rank}'))
                    test_model.eval()

                    test_pred = []
                    test_label = []

                    with torch.no_grad():
                        for batch_idx, batch in enumerate(test_loader):
                            batch_g1, batch_g2, batch_lbl = batch
                            batch_g1 = batch_g1.to(device)
                            batch_g2 = batch_g2.to(device)
                            fea1 = batch_g1.ndata['fea'].to(device, dtype=torch.float32)
                            fea2 = batch_g2.ndata['fea'].to(device, dtype=torch.float32)
                            batch_lbl = batch_lbl.to(device)

                            outputs = test_model(batch_g1, batch_g2, fea1, fea2)
                            # prob = torch.sigmoid(outputs).squeeze()
                            prob = torch.softmax(outputs, dim=1)[:, 1]

                            test_pred.extend(prob.cpu().numpy())
                            test_label.extend(batch_lbl.cpu().numpy())

                            # Periodically clear memory during testing
                            if batch_idx % 5 == 0:
                                del batch_g1, batch_g2, fea1, fea2, batch_lbl, outputs, prob
                                torch.cuda.empty_cache()

                    if len(np.unique(test_label)) >= 2:
                        # Compute AUPRC and AUROC
                        precision_curve, recall_curve, _ = precision_recall_curve(test_label, test_pred)
                        test_auprc_best_auroc = auc(recall_curve, precision_curve)
                        test_auroc_best_auroc = roc_auc_score(test_label, test_pred)

                        # Compute other evaluation metrics
                        test_pred_binary = np.array(test_pred) > 0.5
                        test_acc = accuracy_score(test_label, test_pred_binary)
                        test_precision = precision_score(test_label, test_pred_binary, zero_division=0)
                        test_recall = recall_score(test_label, test_pred_binary, zero_division=0)
                        test_f1 = f1_score(test_label, test_pred_binary, zero_division=0)
                        test_mcc = matthews_corrcoef(test_label, test_pred_binary)
                        test_cm = confusion_matrix(test_label, test_pred_binary)

                        # Save test metrics for best AUROC model
                        fold_metrics['test_acc_best_auroc'] = test_acc
                        fold_metrics['test_precision_best_auroc'] = test_precision
                        fold_metrics['test_recall_best_auroc'] = test_recall
                        fold_metrics['test_f1_best_auroc'] = test_f1
                        fold_metrics['test_mcc_best_auroc'] = test_mcc

                        print(f"📈 Best AUROC model on test set:")
                        print(f"   AUPRC={test_auprc_best_auroc:.4f}, AUROC={test_auroc_best_auroc:.4f}")
                        print(f"   ACC={test_acc:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, MCC={test_mcc:.4f}")
                        print(f"   Confusion Matrix:\n{test_cm}")

                        # Save test results for best AUROC model
                        test_results_file = fold_paths['test_results_auroc']
                        with open(test_results_file, "w", encoding='utf-8') as f:
                            f.write(f"=== Fold {fold} Test Results - Best AUROC Model ===\n")
                            f.write(f"Model path: {best_auroc_model_path}\n")
                            f.write(f"Number of test samples: {len(test_label)}\n")
                            f.write(f"Number of positive samples: {sum(test_label)}\n")
                            f.write(f"Number of negative samples: {len(test_label) - sum(test_label)}\n\n")
                            f.write(f"Evaluation metrics:\n")
                            f.write(f"AUPRC: {test_auprc_best_auroc:.6f}\n")
                            f.write(f"AUROC: {test_auroc_best_auroc:.6f}\n")
                            f.write(f"Accuracy: {test_acc:.6f}\n")
                            f.write(f"Precision: {test_precision:.6f}\n")
                            f.write(f"Recall: {test_recall:.6f}\n")
                            f.write(f"F1-Score: {test_f1:.6f}\n")
                            f.write(f"MCC: {test_mcc:.6f}\n")
                            f.write(f"Confusion Matrix:\n{test_cm}\n")
                        print(f"✅ Best AUROC model test results saved to: {test_results_file}")
                    else:
                        print("⚠️ Test set contains only one class; evaluation metrics cannot be computed")

                except Exception as e:
                    print(f"❌ Error while evaluating best AUROC model: {e}")
            else:
                test_auprc_best_auroc = test_auprc_best_auprc
                test_auroc_best_auroc = test_auroc_best_auprc
                print(
                    f"📋 Best AUPRC and AUROC models are the same; test results: AUPRC={test_auprc_best_auroc:.4f}, AUROC={test_auroc_best_auroc:.4f}")

                # Create a copy of AUROC results file for the same model
                try:
                    import shutil
                    auprc_file = fold_paths['test_results_auprc']
                    auroc_file = fold_paths['test_results_auroc']
                    if os.path.exists(auprc_file):
                        shutil.copy2(auprc_file, auroc_file)
                        print(f"✅ Copied test results file: {auroc_file}")
                except Exception as e:
                    print(f"⚠️ Failed to copy test results file: {e}")

            # Clean memory between model evaluations
            torch.cuda.empty_cache()
            clear_memory()

            # Evaluate best F1 model
            if os.path.exists(best_f1_model_path):
                try:
                    print(f"📊 Evaluating best F1 model: {best_f1_model_path}")
                    with open(best_f1_model_path, 'rb') as f:
                        test_model.load_state_dict(torch.load(f, map_location=f'cuda:{rank}'))
                    test_model.eval()

                    test_pred = []
                    test_label = []

                    with torch.no_grad():
                        for batch_idx, batch in enumerate(test_loader):
                            batch_g1, batch_g2, batch_lbl = batch
                            batch_g1 = batch_g1.to(device)
                            batch_g2 = batch_g2.to(device)
                            fea1 = batch_g1.ndata['fea'].to(device, dtype=torch.float32)
                            fea2 = batch_g2.ndata['fea'].to(device, dtype=torch.float32)
                            batch_lbl = batch_lbl.to(device)

                            outputs = test_model(batch_g1, batch_g2, fea1, fea2)
                            prob = torch.softmax(outputs, dim=1)[:, 1]

                            test_pred.extend(prob.cpu().numpy())
                            test_label.extend(batch_lbl.cpu().numpy())

                            if batch_idx % 5 == 0:
                                del batch_g1, batch_g2, fea1, fea2, batch_lbl, outputs, prob
                                torch.cuda.empty_cache()

                    if len(np.unique(test_label)) >= 2:
                        precision_curve, recall_curve, _ = precision_recall_curve(test_label, test_pred)
                        test_auprc_best_f1 = auc(recall_curve, precision_curve)
                        test_auroc_best_f1 = roc_auc_score(test_label, test_pred)

                        test_pred_binary = np.array(test_pred) > 0.5
                        test_acc = accuracy_score(test_label, test_pred_binary)
                        test_precision = precision_score(test_label, test_pred_binary, zero_division=0)
                        test_recall = recall_score(test_label, test_pred_binary, zero_division=0)
                        test_f1 = f1_score(test_label, test_pred_binary, zero_division=0)
                        test_mcc = matthews_corrcoef(test_label, test_pred_binary)
                        test_cm = confusion_matrix(test_label, test_pred_binary)

                        fold_metrics['test_acc_best_f1'] = test_acc
                        fold_metrics['test_precision_best_f1'] = test_precision
                        fold_metrics['test_recall_best_f1'] = test_recall
                        fold_metrics['test_f1_best_f1'] = test_f1
                        fold_metrics['test_mcc_best_f1'] = test_mcc

                        print(f"📈 Best F1 model on test set:")
                        print(f"   AUPRC={test_auprc_best_f1:.4f}, AUROC={test_auroc_best_f1:.4f}")
                        print(f"   ACC={test_acc:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, MCC={test_mcc:.4f}")
                        print(f"   Confusion Matrix:\n{test_cm}")

                        test_results_file = fold_paths['test_results_f1']
                        with open(test_results_file, "w", encoding='utf-8') as f:
                            f.write(f"=== Fold {fold} Test Results - Best F1 Model ===\n")
                            f.write(f"Model path: {best_f1_model_path}\n")
                            f.write(f"Number of test samples: {len(test_label)}\n")
                            f.write(f"Number of positive samples: {sum(test_label)}\n")
                            f.write(f"Number of negative samples: {len(test_label) - sum(test_label)}\n\n")
                            f.write(f"Evaluation metrics:\n")
                            f.write(f"AUPRC: {test_auprc_best_f1:.6f}\n")
                            f.write(f"AUROC: {test_auroc_best_f1:.6f}\n")
                            f.write(f"Accuracy: {test_acc:.6f}\n")
                            f.write(f"Precision: {test_precision:.6f}\n")
                            f.write(f"Recall: {test_recall:.6f}\n")
                            f.write(f"F1-Score: {test_f1:.6f}\n")
                            f.write(f"MCC: {test_mcc:.6f}\n")
                            f.write(f"Confusion Matrix:\n{test_cm}\n")
                        print(f"✅ Best F1 model test results saved to: {test_results_file}")
                    else:
                        print("⚠️ Test set contains only one class; evaluation metrics cannot be computed")

                except Exception as e:
                    print(f"❌ Error while evaluating best F1 model: {e}")

            # Clean test model after evaluation
            print("🧹 Cleaning test model...")
            del test_model, test_loader
            torch.cuda.empty_cache()
            clear_memory()
            print(f"GPU memory usage after testing: {get_memory_usage()}")

        # Record training completion
        try:
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"=== Training Completed - Fold {fold} ===\n")
                f.write(f"Best AUPRC: {best_auprc:.4f} @ Epoch {best_auprc_epoch}\n")
                f.write(f"Best AUROC: {best_auroc:.4f} @ Epoch {best_auroc_epoch}\n")
                f.write(f"Best F1: {best_f1:.4f} @ Epoch {best_f1_epoch}\n")
                f.write(f"Test set - Best AUPRC model: AUPRC={test_auprc_best_auprc:.4f}, AUROC={test_auroc_best_auprc:.4f}\n")
                f.write(f"Test set - Best AUROC model: AUPRC={test_auprc_best_auroc:.4f}, AUROC={test_auroc_best_auroc:.4f}\n")
                f.write(f"Test set - Best F1 model: AUPRC={test_auprc_best_f1:.4f}, AUROC={test_auroc_best_f1:.4f}\n")
        except Exception as e:
            print(f"⚠️ Failed to write training-completion log: {e}")

        print(f"✅ Fold {fold} training completed - AUPRC: {best_auprc:.4f}, AUROC: {best_auroc:.4f}, F1: {best_f1:.4f}")

        # Save final results (update existing dict to avoid overwriting earlier metrics)
        fold_metrics.update({
            'fold': fold,
            'best_auprc': best_auprc,
            'best_auprc_epoch': best_auprc_epoch,
            'best_auroc': best_auroc,
            'best_auroc_epoch': best_auroc_epoch,
            'best_f1': best_f1,
            'best_f1_epoch': best_f1_epoch,
            'best_auprc_model_path': best_auprc_model_path,
            'best_auroc_model_path': best_auroc_model_path,
            'best_f1_model_path': best_f1_model_path,
            'test_auprc_best_auprc': test_auprc_best_auprc,
            'test_auroc_best_auprc': test_auroc_best_auprc,
            'test_auprc_best_auroc': test_auprc_best_auroc,
            'test_auroc_best_auroc': test_auroc_best_auroc,
            'test_auprc_best_f1': test_auprc_best_f1,
            'test_auroc_best_f1': test_auroc_best_f1,
        })

        if rank == 0:
            try:
                import json
                fold_result_file = fold_paths['fold_result']
                with open(fold_result_file, 'w') as f:
                    json.dump(fold_metrics, f, indent=2)
                print(f"📁 Fold {fold} results saved to {fold_result_file}")
            except Exception as e:
                print(f"⚠️ Unable to save JSON results for Fold {fold}: {e}")

        if rank == 0:
            try:
                with open(commons['kfold_summary'], "a", encoding='utf-8') as f:
                    f.write(f"Fold {fold}: "
                            f"Best_AUPRC={best_auprc:.4f}@{best_auprc_epoch}, "
                            f"Best_AUROC={best_auroc:.4f}@{best_auroc_epoch}, "
                            f"Val_ACC={fold_metrics['best_val_acc']:.4f}, "
                            f"Val_Precision={fold_metrics['best_val_precision']:.4f}, "
                            f"Val_Recall={fold_metrics['best_val_recall']:.4f}, "
                            f"Val_F1={fold_metrics['best_val_f1']:.4f}, "
                            f"Val_MCC={fold_metrics['best_val_mcc']:.4f}, "
                            f"Test_AUPRC_BestAUPRC={test_auprc_best_auprc:.4f}, "
                            f"Test_AUROC_BestAUPRC={test_auroc_best_auprc:.4f}, "
                            f"Test_ACC_BestAUPRC={fold_metrics['test_acc_best_auprc']:.4f}, "
                            f"Test_Precision_BestAUPRC={fold_metrics['test_precision_best_auprc']:.4f}, "
                            f"Test_Recall_BestAUPRC={fold_metrics['test_recall_best_auprc']:.4f}, "
                            f"Test_F1_BestAUPRC={fold_metrics['test_f1_best_auprc']:.4f}, "
                            f"Test_MCC_BestAUPRC={fold_metrics['test_mcc_best_auprc']:.4f}, "
                            f"Test_AUPRC_BestAUROC={test_auprc_best_auroc:.4f}, "
                            f"Test_AUROC_BestAUROC={test_auroc_best_auroc:.4f}, "
                            f"Test_ACC_BestAUROC={fold_metrics['test_acc_best_auroc']:.4f}, "
                            f"Test_Precision_BestAUROC={fold_metrics['test_precision_best_auroc']:.4f}, "
                            f"Test_Recall_BestAUROC={fold_metrics['test_recall_best_auroc']:.4f}, "
                            f"Test_F1_BestAUROC={fold_metrics['test_f1_best_auroc']:.4f}, "
                            f"Test_MCC_BestAUROC={fold_metrics['test_mcc_best_auroc']:.4f}\n")
            except Exception as e:
                print(f"⚠️ Unable to write result summary: {e}")

        # After testing and saving, synchronize all processes before exit to reduce NCCL shutdown noise
        try:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                if rank == 0:
                    print(f"🧩 Fold {fold} testing and saving completed, all processes synchronized and exiting")
        except Exception as e:
            print(f"⚠️ Rank {rank}: synchronization failed at end of training: {e}")

    except OSError as e:
        print(f"❌ Fold {fold} encountered an I/O error: {e}")
        try:
            with open(f"imbalance_human/imbalance_human-error_fold{fold}.txt", "w") as f:
                f.write(f"Training failed: {e}\n")
                f.write(f"Error type: {type(e).__name__}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
        except:
            pass
        raise

    except Exception as e:
        print(f"❌ Fold {fold} encountered an unexpected error: {e}")
        print(f"Error details: {traceback.format_exc()}")
        try:
            with open(f"imbalance_human/imbalance_human-error_fold{fold}_detailed.txt", "w") as f:
                f.write(f"Detailed error information:\n")
                f.write(f"Exception type: {type(e).__name__}\n")
                f.write(f"Exception message: {str(e)}\n")
                f.write(f"Stack trace:\n{traceback.format_exc()}\n")
        except:
            pass
        raise

    finally:
        # 🚀 Multi-GPU safe cleanup: ensure resources are released correctly
        try:
            # Clean up DataLoader
            if 'train_loader' in locals():
                safe_cleanup_dataloaders(train_loader)
            if 'val_loader' in locals():
                safe_cleanup_dataloaders(val_loader)
            if 'test_loader' in locals():
                safe_cleanup_dataloaders(test_loader)
        except Exception as cleanup_e:
            print(f"⚠️ Rank {rank}: DataLoader cleanup failed: {cleanup_e}")
        
        try:
            # Clean CUDA cache
            clear_memory()
        except Exception as mem_e:
            print(f"⚠️ Rank {rank}: memory cleanup failed: {mem_e}")
        
        try:
            # Clean distributed process group
            cleanup()
        except Exception as dist_e:
            print(f"⚠️ Rank {rank}: distributed cleanup failed: {dist_e}")
            print(f"⚠️ Distributed cleanup failed: {e}")

def save_fold_data_splits(fold, train_idx, val_idx, test_idx, ds_all):
    """Save each fold's train/val/test data content to txt files."""
    try:
        # Create output directory (centralized path)
        output_dir = common_paths()['fold_data_splits_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Get sample data
        def get_samples_by_indices(indices):
            samples = []
            for idx in indices:
                # Find the batch file and position
                file_idx = np.searchsorted(ds_all.cum_sizes, idx, 'right') - 1
                pos = idx - ds_all.cum_sizes[file_idx]

                # Load the batch file
                with open(ds_all.batch_files[file_idx], "rb") as f:
                    batch = pickle.load(f)

                sample = batch[pos]
                samples.append(sample)
                del batch  # Release memory immediately
            return samples

        # Save training set
        train_file = os.path.join(output_dir, f"fold_{fold}_train.txt")
        train_samples = get_samples_by_indices(train_idx)
        with open(train_file, "w", encoding='utf-8') as f:
            f.write("# Training set data - format: protein1_id\tprotein2_id\tlabel\n")
            for p1, p2, label in train_samples:
                f.write(f"{p1}\t{p2}\t{label}\n")
        print(f"✅ Training set saved to: {train_file} ({len(train_samples)} samples)")

        # Save validation set
        val_file = os.path.join(output_dir, f"fold_{fold}_val.txt")
        val_samples = get_samples_by_indices(val_idx)
        with open(val_file, "w", encoding='utf-8') as f:
            f.write("# Validation set data - format: protein1_id\tprotein2_id\tlabel\n")
            for p1, p2, label in val_samples:
                f.write(f"{p1}\t{p2}\t{label}\n")
        print(f"✅ Validation set saved to: {val_file} ({len(val_samples)} samples)")

        # Save test set
        test_file = os.path.join(output_dir, f"fold_{fold}_test.txt")
        test_samples = get_samples_by_indices(test_idx)
        with open(test_file, "w", encoding='utf-8') as f:
            f.write("# Test set data - format: protein1_id\tprotein2_id\tlabel\n")
            for p1, p2, label in test_samples:
                f.write(f"{p1}\t{p2}\t{label}\n")
        print(f"✅ Test set saved to: {test_file} ({len(test_samples)} samples)")

        # Cleanup memory
        del train_samples, val_samples, test_samples
        gc.collect()

    except Exception as e:
        print(f"❌ Failed to save data split for fold {fold}: {e}")


def process_pair(p1, p2, label):
    """Validate protein pair loading and record detailed failure reasons."""
    try:
        # Check protein existence
        if not data_use_imbalance.check_protein_exists(p1):
            msg = f"Protein {p1} is not in the dataset"
            with open("failed_pairs_details.txt", "a") as f:
                f.write(f"Failed sample: ({p1}, {p2}), label: {label}, reason: {msg}\n")
            return ("fail", msg)
        if not data_use_imbalance.check_protein_exists(p2):
            msg = f"Protein {p2} is not in the dataset"
            with open("failed_pairs_details.txt", "a") as f:
                f.write(f"Failed sample: ({p1}, {p2}), label: {label}, reason: {msg}\n")
            return ("fail", msg)

        # Try to load graph data
        try:
            temp_g1 = data_use_imbalance.get_protein_graph(p1)
            temp_g2 = data_use_imbalance.get_protein_graph(p2)
        except Exception as e:
            msg = f"Failed to load graph data: {str(e)}"
            with open("failed_pairs_details.txt", "a") as f:
                f.write(f"Failed sample: ({p1}, {p2}), label: {label}, reason: {msg}\n")
            return ("fail", msg)

        # Validate graph data
        if temp_g1.num_nodes() == 0:
            msg = f"Protein {p1} has zero graph nodes"
            with open("failed_pairs_details.txt", "a") as f:
                f.write(f"Failed sample: ({p1}, {p2}), label: {label}, reason: {msg}\n")
            return ("fail", msg)
        if temp_g2.num_nodes() == 0:
            msg = f"Protein {p2} has zero graph nodes"
            with open("failed_pairs_details.txt", "a") as f:
                f.write(f"Failed sample: ({p1}, {p2}), label: {label}, reason: {msg}\n")
            return ("fail", msg)

        if 'fea' not in temp_g1.ndata:
            msg = f"Protein {p1} graph is missing 'fea' node features"
            with open("failed_pairs_details.txt", "a") as f:
                f.write(f"Failed sample: ({p1}, {p2}), label: {label}, reason: {msg}\n")
            return ("fail", msg)
        if 'fea' not in temp_g2.ndata:
            msg = f"Protein {p2} graph is missing 'fea' node features"
            with open("failed_pairs_details.txt", "a") as f:
                f.write(f"Failed sample: ({p1}, {p2}), label: {label}, reason: {msg}\n")
            return ("fail", msg)

        return ("success", (p1, p2, label))
    except Exception as e:
        msg = f"Processing failed: {str(e)}"
        with open("failed_pairs_details.txt", "a") as f:
            f.write(f"Failed sample: ({p1}, {p2}), label: {label}, reason: {msg}\n")
        return ("fail", msg)

def main():
    import json
    from sklearn.model_selection import train_test_split, StratifiedKFold

    print("🔍 Starting training...")
    # Quick environment check
    try:
        print(f"CUDA available (torch): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA current device: {torch.cuda.current_device()}")
        # DGL CUDA capability: DGL 2.4+ lacks dgl.cuda.is_available(); use a minimal operator self-test
        def _dgl_cuda_self_test():
            if not torch.cuda.is_available():
                return False, "torch.cuda not available"
            try:
                g = dgl.graph((torch.tensor([0,1]), torch.tensor([1,0])), num_nodes=2)
                g = g.to('cuda')
                x = torch.randn(2, 3, device='cuda')
                conv = dgl.nn.GraphConv(3, 4).to('cuda')
                y = conv(g, x)
                return bool(getattr(y, 'is_cuda', y.device.type == 'cuda')), None
            except Exception as e:
                return False, str(e)

        dgl_ok, dgl_err = _dgl_cuda_self_test()
        print(f"CUDA available (dgl, self-test): {dgl_ok}")
        if torch.cuda.is_available() and not dgl_ok:
            print("⚠️ Notice: DGL CUDA self-test failed; graph operators may run on CPU. Reason:", dgl_err)
            if os.getenv('PPI_ENFORCE_DGL_CUDA', '0') == '1':
                raise SystemExit("DGL CUDA not available and strict mode enabled (PPI_ENFORCE_DGL_CUDA=1); exiting to avoid CPU training")
    except Exception as _e:
        print(f"Environment-check print exception: {_e}")
    print(f"Initial memory: {get_memory_usage()}")

    # Create required output directories (use OUTPUT_ROOT only)
    ensure_output_dirs()
    print(f"✅ Output root directory: {OUTPUT_ROOT}")
    print(f"✅ Split directory: {OUTPUT_ROOT / 'fold_data_splits'}")

    # Load unified dataset
    print("🔄 Loading unified dataset...")
    dataset_meta = data_use_imbalance.load_unified_dataset()
    mode = dataset_meta.get('mode', 'unknown')
    count = dataset_meta.get('count', '?')
    root = dataset_meta.get('root', 'N/A')
    if mode == 'npz':
        print(f"✅ Dataset loaded: using NPZ lazy loading, {count} protein graphs, directory {root}")
    else:
        print(f"✅ Dataset loaded: using PKL compatibility mode, {count} protein graphs, source {root}")
    print(f"   Current memory usage: {get_memory_usage()}")

    # 1) Set graph/coord cache sizes - read env vars with conservative defaults
    main_graph_cache_size = int(os.getenv("PPI_GRAPH_CACHE_SIZE", "2048"))
    main_coord_cache_size = int(os.getenv("PPI_COORD_CACHE_SIZE", "20480"))
    data_use_imbalance.init_graph_cache(max_size=main_graph_cache_size)  # Use a smaller cache in the main process
    data_use_imbalance.init_coord_cache(max_size=main_coord_cache_size)
    print(f"Main process cache -> Graph: {main_graph_cache_size}, Coord: {main_coord_cache_size}")
    if main_graph_cache_size < 512:
        print("⚠️ Recommended: increase PPI_GRAPH_CACHE_SIZE to 1024+ to reduce repeated graph-build overhead")
    
    # 🚀 Optimal strategy: main process preloads to disk cache once, folds share access
    print("🎯 Using hybrid cache strategy: main process fully preloads to disk and fold processes share access")

    # 2) Build sample batches with multithreading and save to disk (includes preloading)
    build_and_save_samples()

    # 3) Use LazySampleDataset to lazily load all samples and extract labels for stratified sampling
    ds_all = LazySampleDataset(SAMPLES_BATCH_DIR)
    labels = []
    for batch_file in ds_all.batch_files:
        with open(batch_file, "rb") as f:
            batch = pickle.load(f)
        labels.extend([sample[2] for sample in batch])
        del batch 
        gc.collect() 
    labels = np.array(labels)

    ds_all.clear_cache()

    n_samples = len(ds_all)
    all_indices = np.arange(n_samples)
    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    print(f"Total samples={n_samples}, trainval={len(trainval_idx)}, test={len(test_idx)}")

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    all_metrics = []

    world_size = max(1, int(os.getenv("PPI_WORLD_SIZE", "3")))
    print(f"🔧 Multi-GPU training enabled (world_size={world_size})")

    for fold, (tr_local, val_local) in enumerate(
            skf.split(trainval_idx, labels[trainval_idx]), 1):

        train_idx = trainval_idx[tr_local]
        val_idx   = trainval_idx[val_local]

        print(f"\n===== Fold {fold}/{K_FOLDS} =====")
        print(f"  train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

        print(f"📁 Saving data split for Fold {fold}...")
        save_fold_data_splits(fold, train_idx, val_idx, test_idx, ds_all)

        try:
            import datetime
            status_file = paths_for_fold(fold)['status']
            with open(status_file, "w", encoding='utf-8') as f:
                f.write(f"=== Fold {fold} Status Record ===\n")
                f.write(f"Start time: {datetime.datetime.now()}\n")
                f.write("Status: launching training process\n")
                f.write(f"Train samples: {len(train_idx)}\n")
                f.write(f"Validation samples: {len(val_idx)}\n")
                f.write(f"Test samples: {len(test_idx)}\n")
        except Exception as e:
            print(f"⚠️ Unable to write status file: {e}")

        preferred_port = BASE_PORT + (fold - 1)
        port = find_free_port(preferred_port)
        if port != preferred_port:
            print(f"⚠️ Fold {fold}: port {preferred_port} is occupied, switching to {port}")
        else:
            print(f"🛰️ Fold {fold}: using port {port}")
        cleanup() 
        
        gc.collect()
        torch.cuda.empty_cache()

        try:
            print(f"🚀 Starting Fold {fold} training...")
            mp.spawn(
                train,
                args=(world_size, fold, port, train_idx, val_idx, test_idx),
                nprocs=world_size,
                join=True
            )
            print(f"✅ Fold {fold} training finished")

            try:
                import datetime
                status_file = paths_for_fold(fold)['status']
                with open(status_file, "a", encoding='utf-8') as f:
                    f.write(f"Completion time: {datetime.datetime.now()}\n")
                    f.write("Status: training completed successfully\n")
            except Exception as e:
                print(f"⚠️ Unable to update status file: {e}")
        except Exception as e:
            error_msg = f"❌ Fold {fold} training failed: {e}"
            print(error_msg)
            print(f"Error details: {traceback.format_exc()}")

            try:
                import datetime
                error_file = paths_for_fold(fold)['error_main']
                with open(error_file, "w", encoding='utf-8') as f:
                    f.write(f"=== Fold {fold} Main Error Report ===\n")
                    f.write(f"Time: {datetime.datetime.now()}\n")
                    f.write(f"Error type: {type(e).__name__}\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write("Possible causes: system sleep, forced shutdown, out-of-memory, GPU errors, etc.\n")
                    f.write(f"Detailed traceback:\n{traceback.format_exc()}\n")
                print(f"📝 Error information written to: {error_file}")
            except Exception as write_error:
                print(f"⚠️ Unable to write error file: {write_error}")

            continue

        gc.collect()
        torch.cuda.empty_cache()

        result_file = paths_for_fold(fold)['fold_result']
        if os.path.exists(result_file):
            with open(result_file) as f:
                fold_metrics = json.load(f)
        else:
            fold_metrics = {
                'fold': fold,
                'best_auprc': 0.0, 'best_auprc_epoch': 0,
                'best_auroc': 0.0, 'best_auroc_epoch': 0,
                'test_auprc_best_auprc': 0.0, 'test_auroc_best_auprc': 0.0,
                'test_auprc_best_auroc': 0.0, 'test_auroc_best_auroc': 0.0,
                'best_val_acc': 0.0, 'best_val_precision': 0.0, 'best_val_recall': 0.0,
                'best_val_f1': 0.0, 'best_val_mcc': 0.0,
                'test_acc_best_auprc': 0.0, 'test_precision_best_auprc': 0.0,
                'test_recall_best_auprc': 0.0, 'test_f1_best_auprc': 0.0, 'test_mcc_best_auprc': 0.0,
                'test_acc_best_auroc': 0.0, 'test_precision_best_auroc': 0.0,
                'test_recall_best_auroc': 0.0, 'test_f1_best_auroc': 0.0, 'test_mcc_best_auroc': 0.0,
                'test_acc_best_f1': 0.0, 'test_precision_best_f1': 0.0,
                'test_recall_best_f1': 0.0, 'test_f1_best_f1': 0.0, 'test_mcc_best_f1': 0.0
            }
        all_metrics.append(fold_metrics)

    print("\n===== K-Fold Cross Validation Results Summary =====")
    print("Validation results:")
    for metric in all_metrics:
        print(f"Fold {metric['fold']}: "
              f"Best AUPRC={metric.get('best_auprc', 0.0):.4f}@{metric.get('best_auprc_epoch', 0)}, "
              f"Best AUROC={metric.get('best_auroc', 0.0):.4f}@{metric.get('best_auroc_epoch', 0)}, "
              f"ACC={metric.get('best_val_acc', 0.0):.4f}, "
              f"Precision={metric.get('best_val_precision', 0.0):.4f}, "
              f"Recall={metric.get('best_val_recall', 0.0):.4f}, "
              f"F1={metric.get('best_val_f1', 0.0):.4f}, "
              f"MCC={metric.get('best_val_mcc', 0.0):.4f}")

    print("\nTest results:")
    for metric in all_metrics:
        print(f"Fold {metric['fold']}: "
              f"Test AUPRC(BestAUPRC)={metric.get('test_auprc_best_auprc', 0.0):.4f}, "
              f"Test AUROC(BestAUPRC)={metric.get('test_auroc_best_auprc', 0.0):.4f}, "
              f"Test ACC(BestAUPRC)={metric.get('test_acc_best_auprc', 0.0):.4f}, "
              f"Test F1(BestAUPRC)={metric.get('test_f1_best_auprc', 0.0):.4f}, "
              f"Test MCC(BestAUPRC)={metric.get('test_mcc_best_auprc', 0.0):.4f}")
        print(f"        "
              f"Test AUPRC(BestAUROC)={metric.get('test_auprc_best_auroc', 0.0):.4f}, "
              f"Test AUROC(BestAUROC)={metric.get('test_auroc_best_auroc', 0.0):.4f}, "
              f"Test ACC(BestAUROC)={metric.get('test_acc_best_auroc', 0.0):.4f}, "
              f"Test F1(BestAUROC)={metric.get('test_f1_best_auroc', 0.0):.4f}, "
              f"Test MCC(BestAUROC)={metric.get('test_mcc_best_auroc', 0.0):.4f}")
        print(f"        "
              f"Test AUPRC(BestF1)={metric.get('test_auprc_best_f1', 0.0):.4f}, "
              f"Test AUROC(BestF1)={metric.get('test_auroc_best_f1', 0.0):.4f}, "
              f"Test ACC(BestF1)={metric.get('test_acc_best_f1', 0.0):.4f}, "
              f"Test F1(BestF1)={metric.get('test_f1_best_f1', 0.0):.4f}, "
              f"Test MCC(BestF1)={metric.get('test_mcc_best_f1', 0.0):.4f}")

    print(f"\nOverall validation performance:")
    print(f"Mean AUPRC: {np.mean([m.get('best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_auprc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean AUROC: {np.mean([m.get('best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_auroc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean ACC: {np.mean([m.get('best_val_acc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_acc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Precision: {np.mean([m.get('best_val_precision', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_precision', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Recall: {np.mean([m.get('best_val_recall', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_recall', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean F1: {np.mean([m.get('best_val_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_f1', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean MCC: {np.mean([m.get('best_val_mcc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_mcc', 0.0) for m in all_metrics]):.4f}")

    print(f"\nOverall test performance (based on best AUPRC model):")
    print(f"Mean Test AUPRC: {np.mean([m.get('test_auprc_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auprc_best_auprc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test AUROC: {np.mean([m.get('test_auroc_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auroc_best_auprc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test ACC: {np.mean([m.get('test_acc_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_acc_best_auprc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test Precision: {np.mean([m.get('test_precision_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_precision_best_auprc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test Recall: {np.mean([m.get('test_recall_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_recall_best_auprc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test F1: {np.mean([m.get('test_f1_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_f1_best_auprc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test MCC: {np.mean([m.get('test_mcc_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_mcc_best_auprc', 0.0) for m in all_metrics]):.4f}")

    print(f"\nOverall test performance (based on best AUROC model):")
    print(f"Mean Test AUPRC: {np.mean([m.get('test_auprc_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auprc_best_auroc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test AUROC: {np.mean([m.get('test_auroc_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auroc_best_auroc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test ACC: {np.mean([m.get('test_acc_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_acc_best_auroc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test Precision: {np.mean([m.get('test_precision_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_precision_best_auroc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test Recall: {np.mean([m.get('test_recall_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_recall_best_auroc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test F1: {np.mean([m.get('test_f1_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_f1_best_auroc', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test MCC: {np.mean([m.get('test_mcc_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_mcc_best_auroc', 0.0) for m in all_metrics]):.4f}")

    print(f"\nOverall test performance (based on best F1 model):")
    print(f"Mean Test AUPRC: {np.mean([m.get('test_auprc_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auprc_best_f1', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test AUROC: {np.mean([m.get('test_auroc_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auroc_best_f1', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test ACC: {np.mean([m.get('test_acc_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_acc_best_f1', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test Precision: {np.mean([m.get('test_precision_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_precision_best_f1', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test Recall: {np.mean([m.get('test_recall_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_recall_best_f1', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test F1: {np.mean([m.get('test_f1_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_f1_best_f1', 0.0) for m in all_metrics]):.4f}")
    print(f"Mean Test MCC: {np.mean([m.get('test_mcc_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_mcc_best_f1', 0.0) for m in all_metrics]):.4f}")

    with open(common_paths()['final_results'], "w") as f:
        f.write("===== K-Fold Cross Validation Results Summary =====\n\n")

        f.write("Validation set results:\n")
        for metric in all_metrics:
            f.write(f"Fold {metric['fold']}: "
                    f"Best AUPRC={metric.get('best_auprc', 0.0):.4f}@{metric.get('best_auprc_epoch', 0)}, "
                    f"Best AUROC={metric.get('best_auroc', 0.0):.4f}@{metric.get('best_auroc_epoch', 0)}, "
                    f"ACC={metric.get('best_val_acc', 0.0):.4f}, "
                    f"Precision={metric.get('best_val_precision', 0.0):.4f}, "
                    f"Recall={metric.get('best_val_recall', 0.0):.4f}, "
                    f"F1={metric.get('best_val_f1', 0.0):.4f}, "
                    f"MCC={metric.get('best_val_mcc', 0.0):.4f}\n")

        f.write("\nTest set results (based on best AUPRC model):\n")
        for metric in all_metrics:
            f.write(f"Fold {metric['fold']}: "
                    f"AUPRC={metric.get('test_auprc_best_auprc', 0.0):.4f}, "
                    f"AUROC={metric.get('test_auroc_best_auprc', 0.0):.4f}, "
                    f"ACC={metric.get('test_acc_best_auprc', 0.0):.4f}, "
                    f"Precision={metric.get('test_precision_best_auprc', 0.0):.4f}, "
                    f"Recall={metric.get('test_recall_best_auprc', 0.0):.4f}, "
                    f"F1={metric.get('test_f1_best_auprc', 0.0):.4f}, "
                    f"MCC={metric.get('test_mcc_best_auprc', 0.0):.4f}\n")

        f.write("\nTest set results (based on best AUROC model):\n")
        for metric in all_metrics:
            f.write(f"Fold {metric['fold']}: "
                    f"AUPRC={metric.get('test_auprc_best_auroc', 0.0):.4f}, "
                    f"AUROC={metric.get('test_auroc_best_auroc', 0.0):.4f}, "
                    f"ACC={metric.get('test_acc_best_auroc', 0.0):.4f}, "
                    f"Precision={metric.get('test_precision_best_auroc', 0.0):.4f}, "
                    f"Recall={metric.get('test_recall_best_auroc', 0.0):.4f}, "
                    f"F1={metric.get('test_f1_best_auroc', 0.0):.4f}, "
                    f"MCC={metric.get('test_mcc_best_auroc', 0.0):.4f}\n")

        f.write("\nTest set results (based on best F1 model):\n")
        for metric in all_metrics:
            f.write(f"Fold {metric['fold']}: "
                    f"AUPRC={metric.get('test_auprc_best_f1', 0.0):.4f}, "
                    f"AUROC={metric.get('test_auroc_best_f1', 0.0):.4f}, "
                    f"ACC={metric.get('test_acc_best_f1', 0.0):.4f}, "
                    f"Precision={metric.get('test_precision_best_f1', 0.0):.4f}, "
                    f"Recall={metric.get('test_recall_best_f1', 0.0):.4f}, "
                    f"F1={metric.get('test_f1_best_f1', 0.0):.4f}, "
                    f"MCC={metric.get('test_mcc_best_f1', 0.0):.4f}\n")

        f.write("\nModel paths:\n")
        for metric in all_metrics:
            f.write(f"Fold {metric['fold']}:\n"
                    f"  AUPRC Model: {metric['best_auprc_model_path']}\n"
                    f"  AUROC Model: {metric['best_auroc_model_path']}\n"
                    f"  F1 Model: {metric.get('best_f1_model_path', 'N/A')}\n\n")

        f.write(f"\nOverall validation performance:\n")
        f.write(f"Mean AUPRC: {np.mean([m.get('best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_auprc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean AUROC: {np.mean([m.get('best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_auroc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean ACC: {np.mean([m.get('best_val_acc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_acc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Precision: {np.mean([m.get('best_val_precision', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_precision', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Recall: {np.mean([m.get('best_val_recall', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_recall', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean F1: {np.mean([m.get('best_val_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_f1', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean MCC: {np.mean([m.get('best_val_mcc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('best_val_mcc', 0.0) for m in all_metrics]):.4f}\n")

        f.write(f"\nOverall test performance (based on best AUPRC model):\n")
        f.write(f"Mean Test AUPRC: {np.mean([m.get('test_auprc_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auprc_best_auprc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test AUROC: {np.mean([m.get('test_auroc_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auroc_best_auprc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test ACC: {np.mean([m.get('test_acc_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_acc_best_auprc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test Precision: {np.mean([m.get('test_precision_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_precision_best_auprc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test Recall: {np.mean([m.get('test_recall_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_recall_best_auprc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test F1: {np.mean([m.get('test_f1_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_f1_best_auprc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test MCC: {np.mean([m.get('test_mcc_best_auprc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_mcc_best_auprc', 0.0) for m in all_metrics]):.4f}\n")

        f.write(f"\nOverall test performance (based on best AUROC model):\n")
        f.write(f"Mean Test AUPRC: {np.mean([m.get('test_auprc_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auprc_best_auroc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test AUROC: {np.mean([m.get('test_auroc_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auroc_best_auroc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test ACC: {np.mean([m.get('test_acc_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_acc_best_auroc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test Precision: {np.mean([m.get('test_precision_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_precision_best_auroc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test Recall: {np.mean([m.get('test_recall_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_recall_best_auroc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test F1: {np.mean([m.get('test_f1_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_f1_best_auroc', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test MCC: {np.mean([m.get('test_mcc_best_auroc', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_mcc_best_auroc', 0.0) for m in all_metrics]):.4f}\n")

        f.write(f"\nOverall test performance (based on best F1 model):\n")
        f.write(f"Mean Test AUPRC: {np.mean([m.get('test_auprc_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auprc_best_f1', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test AUROC: {np.mean([m.get('test_auroc_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_auroc_best_f1', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test ACC: {np.mean([m.get('test_acc_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_acc_best_f1', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test Precision: {np.mean([m.get('test_precision_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_precision_best_f1', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test Recall: {np.mean([m.get('test_recall_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_recall_best_f1', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test F1: {np.mean([m.get('test_f1_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_f1_best_f1', 0.0) for m in all_metrics]):.4f}\n")
        f.write(f"Mean Test MCC: {np.mean([m.get('test_mcc_best_f1', 0.0) for m in all_metrics]):.4f} ± {np.std([m.get('test_mcc_best_f1', 0.0) for m in all_metrics]):.4f}\n")

    data_use_imbalance.clear_graph_cache()


if __name__ == "__main__":
    main()