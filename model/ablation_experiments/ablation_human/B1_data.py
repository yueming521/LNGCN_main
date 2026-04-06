import os
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp
import dgl
from sklearn.preprocessing import StandardScaler
import pickle
import h5py
import time
import torch.multiprocessing as mp
from collections import OrderedDict
from Bio import PDB  
from typing import Optional, Callable, List, Tuple, Dict
import fcntl

DEBUG_DATA = os.getenv("PPI_DATA_DEBUG", "0") == "1"
QUIET = os.getenv("PPI_QUIET", "0") == "1"
_raw_coord_cache_max = int(os.getenv("PPI_COORD_CACHE_SIZE", "4096"))
_coord_cache_mode = os.getenv("PPI_COORD_CACHE_MODE", "memory_disk").strip().lower()
if _coord_cache_mode in {"disk", "disk_only", "off", "none"}:
    _raw_coord_cache_max = 0
COORD_CACHE_MAX = max(0, _raw_coord_cache_max)  

COORD_CACHE_READ_DIR = os.getenv(
    "PPI_COORD_CACHE_READ_DIR", 
    "LNGCN_main/results/ablation/ablation_human/B1_NoLTC/pdb_coords_cache_n_B1"
)
COORD_CACHE_SAVE_DIR = os.getenv(
    "PPI_COORD_CACHE_SAVE_DIR", 
    "LNGCN_main/results/ablation/ablation_human/B1_NoLTC/pdb_coords_cache_n_B1"
)
COORD_CACHE_DIR = COORD_CACHE_READ_DIR

ENABLE_COORD_PRELOAD = os.getenv("PPI_ENABLE_COORD_PRELOAD", "1") == "1"
COORD_PRELOAD_THREADS = max(1, int(os.getenv("PPI_COORD_PRELOAD_THREADS", "2")))
COORD_PRELOAD_CHUNK = max(1, int(os.getenv("PPI_COORD_PRELOAD_CHUNK", "1000")))

def _resolve_output_root() -> str:
    if root and len(root.strip()) > 0:
        try:
            os.makedirs(root, exist_ok=True)
            return root
        except Exception:
            pass
    default_root = "LNGCN_main/results/ablation/ablation_human/B1_NoLTC"
    try:
        os.makedirs(default_root, exist_ok=True)
    except Exception:
        pass
    return default_root

def _resolve_results_dir():
    for d in ("3714numpy", "56547"):
        if os.path.isdir(d):
            return d
    return _resolve_output_root()

def _append_fail_pdb(protein_id: str, reason: str):
    results_dir = _resolve_results_dir()
    fail_file = os.path.join(results_dir, "fail-pdb-time.txt")
    try:
        with open(fail_file, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(f"{protein_id}\t{reason}\n")
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                f.write(f"{protein_id}\t{reason}\n")
    except Exception as _e:
        pass  

def append_fail_cache(protein_id: str, reason: str, label: Optional[str] = None):
    out_dir = _resolve_output_root()
    fail_file = os.path.join(out_dir, "failcache.txt")
    line = f"[{label}]\t{protein_id}\t{reason}\n" if label else f"{protein_id}\t{reason}\n"
    try:
        with open(fail_file, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(line)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                f.write(line)
    except Exception as _e:
        pass 


DATA_BACKEND = None 
GRAPH_CACHE = None

CACHE_MAX_SIZE = int(os.getenv("PPI_GRAPH_CACHE_SIZE", "64"))
PDB_DIR = "LNGCN_main/data/data_base/human_balance_all_data/human_balance_pdb" 
GRAPH_NPZ_DIR = os.getenv(
    "PPI_GRAPH_NPZ_DIR",
    "LNGCN_main/data/data_base/human_balance_all_data/human_balance_npz"
)
GRAPH_PKL_PATH = os.getenv(
    "PPI_GRAPH_PKL_PATH",
    "LNGCN_main/data/data_base/human_balance_all_data/human_balance_pkl"
)
PDB_DIR = "18844/18844pdb" 

GRAPH_NPZ_DIR = os.getenv(
    "PPI_GRAPH_NPZ_DIR",
    "18844/18844npz"
)
GRAPH_PKL_PATH = os.getenv(
    "PPI_GRAPH_PKL_PATH",
    "18844/18844all.pkl"
)
COORD_CACHE = None 

def init_graph_cache(max_size: Optional[int] = None):
    global GRAPH_CACHE, CACHE_MAX_SIZE
    if max_size is None:
        max_size = int(os.getenv("PPI_GRAPH_CACHE_SIZE", str(CACHE_MAX_SIZE)))
    CACHE_MAX_SIZE = max(0, int(max_size))
    GRAPH_CACHE = OrderedDict()
    if not QUIET:
        print(f"Initialized single-process graph cache, max size: {CACHE_MAX_SIZE}")


def init_coord_cache(max_size=COORD_CACHE_MAX):
    global COORD_CACHE, COORD_CACHE_MAX
    COORD_CACHE_MAX = max_size
    COORD_CACHE = OrderedDict()
    try:
        os.makedirs(COORD_CACHE_SAVE_DIR, exist_ok=True)
    except Exception:
        pass
    if DEBUG_DATA and not QUIET:
        print(f"Initialized PDB coordinate cache: memory LRU limit={COORD_CACHE_MAX}, save disk cache dir={COORD_CACHE_SAVE_DIR}, read disk cache dir={COORD_CACHE_READ_DIR}")


def _add_to_coord_cache(protein_id: str, arr: np.ndarray):
    global COORD_CACHE
    if COORD_CACHE_MAX <= 0 or COORD_CACHE is None:
        return
    if protein_id in COORD_CACHE:
        COORD_CACHE.move_to_end(protein_id, last=True)
        return
    if len(COORD_CACHE) >= COORD_CACHE_MAX:
        try:
            oldest_id = next(iter(COORD_CACHE))
            del COORD_CACHE[oldest_id]
        except StopIteration:
            pass
    COORD_CACHE[protein_id] = arr
    COORD_CACHE.move_to_end(protein_id, last=True)


def _cache_graph(protein_id, graph):
    global GRAPH_CACHE
    if GRAPH_CACHE is None:
        init_graph_cache()
    if CACHE_MAX_SIZE <= 0:
        return
    if len(GRAPH_CACHE) >= CACHE_MAX_SIZE and protein_id not in GRAPH_CACHE:
        oldest_id = next(iter(GRAPH_CACHE))
        del GRAPH_CACHE[oldest_id]
    GRAPH_CACHE[protein_id] = graph
    GRAPH_CACHE.move_to_end(protein_id, last=True) 

def clear_graph_cache():
    global GRAPH_CACHE
    if GRAPH_CACHE is not None:
        GRAPH_CACHE.clear()
    if not QUIET:
        print("Protein graph cache cleared")

def _build_npz_index(npz_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not npz_dir or not os.path.isdir(npz_dir):
        return mapping
    for entry in os.scandir(npz_dir):
        if entry.is_file() and entry.name.lower().endswith('.npz'):
            protein_id = entry.name[:-4]
            if protein_id.endswith('_graph'):
                protein_id = protein_id[:-6]
            mapping[protein_id] = entry.path
    return mapping


def _load_npz_graph(npz_path: str, protein_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(npz_path, allow_pickle=True) as npz:
        feature_keys = ("features", "node_features", "fea")
        features = None
        for key in feature_keys:
            if key in npz:
                features = np.asarray(npz[key], dtype=np.float32)
                break
        if features is None:
            raise KeyError(f"NPZ({protein_id}) does not contain node features (keys={list(npz.keys())})")
        num_nodes = features.shape[0]
        src = dst = None
        if "edge_index" in npz:
            edge_index = np.asarray(npz["edge_index"])
            if edge_index.shape[0] != 2:
                edge_index = edge_index.reshape(2, -1)
            src = edge_index[0].astype(np.int64)
            dst = edge_index[1].astype(np.int64)
        elif "row" in npz and "col" in npz:
            src = np.asarray(npz["row"], dtype=np.int64)
            dst = np.asarray(npz["col"], dtype=np.int64)
        elif {"adjacency_data", "adjacency_indices", "adjacency_indptr", "adjacency_shape"}.issubset(npz.keys()):
            csr = sp.csr_matrix(
                (
                    npz["adjacency_data"],
                    npz["adjacency_indices"],
                    npz["adjacency_indptr"],
                ),
                shape=tuple(npz["adjacency_shape"]),
            )
            coo = csr.tocoo()
            src = coo.row.astype(np.int64)
            dst = coo.col.astype(np.int64)
            num_nodes = csr.shape[0]
        elif "adjacency" in npz:
            adjacency = np.asarray(npz["adjacency"])
            if adjacency.ndim != 2:
                raise ValueError(f"NPZ({protein_id}) adjacency has invalid dimensions: {adjacency.shape}")
            src, dst = np.nonzero(adjacency)
            src = src.astype(np.int64)
            dst = dst.astype(np.int64)
            num_nodes = adjacency.shape[0]
        else:
            raise KeyError(f"NPZ({protein_id}) did not contain recognizable edge information (keys={list(npz.keys())})")

        if "num_nodes" in npz:
            num_nodes = int(npz["num_nodes"])

    if src is None or dst is None:
        raise RuntimeError(f"NPZ({protein_id}) failed to parse edge information")

    return features, src, dst, num_nodes


def load_unified_dataset(force_reload: bool = False):
    global DATA_BACKEND
    if DATA_BACKEND is not None and not force_reload:
        return DATA_BACKEND

    npz_mapping = _build_npz_index(GRAPH_NPZ_DIR)
    if npz_mapping:
        DATA_BACKEND = {
            "mode": "npz",
            "root": GRAPH_NPZ_DIR,
            "id_to_path": npz_mapping,
            "count": len(npz_mapping),
        }
        if DEBUG_DATA and not QUIET:
            print(f"[dataset] NPZ backend loaded: {len(npz_mapping)} proteins from {GRAPH_NPZ_DIR}")
        return DATA_BACKEND

    if not os.path.exists(GRAPH_PKL_PATH):
        raise FileNotFoundError(
            f"NPZ directory ({GRAPH_NPZ_DIR}) or PKL file ({GRAPH_PKL_PATH}) not found; please check the path"
        )
    if DEBUG_DATA and not QUIET:
        print(f"[dataset] NPZ not ready, falling back to PKL: {GRAPH_PKL_PATH}")

    with open(GRAPH_PKL_PATH, "rb") as f:
        data = pickle.load(f)

    graphs = data.get("graphs")
    if graphs is None:
        raise KeyError(f"PKL({GRAPH_PKL_PATH}) does not contain the 'graphs' key")

    id_to_index = {
        graph.get("protein_id", f"idx_{idx}"): idx
        for idx, graph in enumerate(graphs)
    }

    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    DATA_BACKEND = {
        "mode": "pkl",
        "graphs": graphs,
        "id_to_index": id_to_index,
        "root": os.path.dirname(GRAPH_PKL_PATH),
        "count": len(graphs),
    }
    return DATA_BACKEND

def batch_preload_coordinates(protein_ids: List[str], max_workers: int = 4, label: str = "preload", every: int = 1000, quiet: Optional[bool] = None):
    if quiet is None:
        quiet = QUIET
    import time
    global COORD_CACHE
    if COORD_CACHE is None:
        init_coord_cache()

    if not ENABLE_COORD_PRELOAD:
        if not quiet:
            print(f"🛑 [{label}] Coordinate preload is disabled (PPI_ENABLE_COORD_PRELOAD=0), disk cache will be accessed on demand")
        return
    
    disk_cache_hits = 0
    existing_disk_files = set()
    if os.path.exists(COORD_CACHE_READ_DIR):
        existing_disk_files.update(f[:-4] for f in os.listdir(COORD_CACHE_READ_DIR) if f.endswith('.npy'))
    if os.path.exists(COORD_CACHE_SAVE_DIR):
        existing_disk_files.update(f[:-4] for f in os.listdir(COORD_CACHE_SAVE_DIR) if f.endswith('.npy'))
    disk_cache_hits = len(set(protein_ids) & existing_disk_files)
    
    unique_ids = list(set(protein_ids))
    if COORD_CACHE_MAX > 0:
        to_load = [pid for pid in unique_ids if pid not in COORD_CACHE]
    else:
        to_load = unique_ids
    total = len(unique_ids)
    memory_hits = (len(unique_ids) - len(to_load)) if COORD_CACHE_MAX > 0 else 0
    
    if not quiet:
        print(f"📊 [{label}] Coordinate cache status:")
        print(f"  Target proteins: {total}")
        print(f"  Memory hits: {memory_hits} | Disk hits (available): {disk_cache_hits}")
        print(f"  Estimated PDB files to parse: {max(0, len(to_load) - disk_cache_hits)}")
    
    if not to_load:
        if not quiet:
            print(f"✅ [{label}] All {total} protein coordinates are already in memory cache")
        return
        
    if not quiet:
        print(f"🚀 [{label}] Starting batch preload of {len(to_load)} protein coordinates...")
    
    def load_single_coord(protein_id):
        try:
            start_time = time.time()
            coords = get_calpha_coordinates(protein_id)
            load_time = time.time() - start_time
            return protein_id, True, coords.shape, load_time
        except Exception as e:
            append_fail_cache(protein_id, str(e), label)
            return protein_id, False, str(e), 0
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    success_count = 0
    total_load_time = 0
    batch_start = time.time()
    processed = 0
    
    actual_workers = max(1, min(max_workers, COORD_PRELOAD_THREADS))
    
    BATCH_CHUNK_SIZE = COORD_PRELOAD_CHUNK
    
    for chunk_start in range(0, len(to_load), BATCH_CHUNK_SIZE):
        chunk_end = min(chunk_start + BATCH_CHUNK_SIZE, len(to_load))
        chunk_proteins = to_load[chunk_start:chunk_end]
        
        if not quiet:
            print(f"🔄 [{label}] Processing chunk {chunk_start//BATCH_CHUNK_SIZE + 1}/{(len(to_load)-1)//BATCH_CHUNK_SIZE + 1}: "
                f"{len(chunk_proteins)} proteins")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(load_single_coord, pid): pid for pid in chunk_proteins}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc=f"[{label}] chunk{chunk_start//BATCH_CHUNK_SIZE + 1}", unit="protein", disable=quiet):
                protein_id, success, info, load_time = future.result()
                if success:
                    success_count += 1
                    total_load_time += load_time
                else:
                    if DEBUG_DATA and not quiet:
                        print(f"❌ Preload failed for {protein_id}: {info}")
                processed += 1
                if processed % max(1, every) == 0 or processed == len(to_load):
                    if not quiet:
                        elapsed = time.time() - batch_start
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        remain = len(to_load) - processed
                        eta = remain / rate if rate > 0 else float('inf')
                        pct = 100.0 * processed / max(1, len(to_load))
                        print(f"⏱️ [{label}] Progress: {processed}/{len(to_load)} ({pct:.1f}%), speed: {rate:.1f}/s, estimated remaining: {eta:.1f}s")
        
        if COORD_CACHE_MAX > 0 and len(COORD_CACHE) > COORD_CACHE_MAX * 0.9:
            if not quiet:
                print(f"⚠️ [{label}] Memory cache is close to its limit ({len(COORD_CACHE)}/{COORD_CACHE_MAX})")
    
    batch_time = time.time() - batch_start
    avg_speed = success_count / batch_time if batch_time > 0 else 0
    
    if not quiet:
        print(f"✅ [{label}] Coordinate preload complete: {success_count}/{len(to_load)} succeeded")
        if COORD_CACHE_MAX > 0:
            print(f"📊 [{label}] Memory cache status: {len(COORD_CACHE)}/{COORD_CACHE_MAX}")
        print(f"⏱️ [{label}] Total elapsed: {batch_time:.1f}s, average speed: {avg_speed:.1f} proteins/s")
        print(f"💾 [{label}] Preload stage finished")

def get_calpha_coordinates(protein_id: str):
    import time
    global COORD_CACHE
    if COORD_CACHE is None:
        init_coord_cache()

    if COORD_CACHE_MAX > 0 and protein_id in COORD_CACHE:
        COORD_CACHE.move_to_end(protein_id, last=True)
        return COORD_CACHE[protein_id]

    read_disk_path = os.path.join(COORD_CACHE_READ_DIR, f"{protein_id}.npy")
    save_disk_path = os.path.join(COORD_CACHE_SAVE_DIR, f"{protein_id}.npy")

    if os.path.exists(read_disk_path):
        try:
            arr = np.load(read_disk_path)
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3:
                _add_to_coord_cache(protein_id, arr)
                if DEBUG_DATA and not QUIET:
                    print(f"[coords-cache] Hit read-disk cache: {protein_id}, shape={arr.shape}, path={read_disk_path}")
                return arr
        except Exception:
            pass

    if os.path.exists(save_disk_path):
        try:
            arr = np.load(save_disk_path)
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3:
                _add_to_coord_cache(protein_id, arr)
                if DEBUG_DATA and not QUIET:
                    print(f"[coords-cache] Hit save-disk cache: {protein_id}, shape={arr.shape}, path={save_disk_path}")
                return arr
        except Exception:
            pass

    t0 = time.time()
    pdb_path = os.path.join(PDB_DIR, f"{protein_id}.pdb")
    if not os.path.exists(pdb_path):
        err = f"PDB file does not exist: {pdb_path}"
        append_fail_cache(protein_id, err, label="coords-parse")
        raise FileNotFoundError(err)

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(protein_id, pdb_path)
    except Exception as e:
        append_fail_cache(protein_id, f"PDB parsing failed: {e}", label="coords-parse")
        raise

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                    else:
                        continue

    if not coords:
        append_fail_cache(protein_id, "Cα atom coordinates not found", label="coords-parse")
        raise ValueError(f"Cα atom coordinates not found: {protein_id}")

    arr = np.array(coords, dtype=np.float32)
    try:
        os.makedirs(COORD_CACHE_SAVE_DIR, exist_ok=True)
        np.save(save_disk_path, arr)
    except Exception as e:
        append_fail_cache(protein_id, f"Failed to write disk cache: {e}", label="coords-cache")
        pass

    try:
        _add_to_coord_cache(protein_id, arr)
    except Exception as e:
        append_fail_cache(protein_id, f"Failed to write memory cache: {e}", label="coords-cache")
        pass

    if DEBUG_DATA and not QUIET:
        print(f"[coords-parse] {protein_id}: residue_count={arr.shape[0]}, elapsed={time.time()-t0:.2f}s")
    return arr


def check_protein_exists_fast(protein_id):
    dataset = load_unified_dataset()
    if dataset['mode'] == 'npz':
        mapping = dataset['id_to_path']
        if protein_id not in mapping:
            return False
    else:
        if protein_id not in dataset['id_to_index']:
            return False
    pdb_path = os.path.join(PDB_DIR, f"{protein_id}.pdb")
    return os.path.exists(pdb_path)

def check_protein_exists(protein_id):
    dataset = load_unified_dataset()
    pdb_exists = os.path.exists(os.path.join(PDB_DIR, f"{protein_id}.pdb"))
    if dataset['mode'] == 'npz':
        return protein_id in dataset['id_to_path'] and pdb_exists
    return protein_id in dataset['id_to_index'] and pdb_exists

def get_protein_graph(protein_id):
    global GRAPH_CACHE
    try:
        from torch.utils.data import get_worker_info as _get_worker_info
        _worker = _get_worker_info()
    except Exception:
        _worker = None
    disable_cache_in_worker = (_worker is not None) and (os.getenv("PPI_WORKER_CACHE", "0") == "0")

    if not disable_cache_in_worker and GRAPH_CACHE is not None and protein_id in GRAPH_CACHE:
        GRAPH_CACHE.move_to_end(protein_id)
        return GRAPH_CACHE[protein_id]

    dataset = load_unified_dataset()
    if dataset['mode'] == 'npz':
        mapping = dataset['id_to_path']
        if protein_id not in mapping:
            raise ValueError(f"{protein_id} is not in NPZ data directory {dataset['root']}")
        features, src_nodes, dst_nodes, num_nodes = _load_npz_graph(mapping[protein_id], protein_id)
    else:
        if protein_id not in dataset['id_to_index']:
            raise ValueError(f"{protein_id} is not in dataset")
        graph_data = dataset['graphs'][dataset['id_to_index'][protein_id]]
        adjacency = graph_data['adjacency']
        if sp.issparse(adjacency):
            coo = adjacency.tocoo()
            src_nodes = coo.row.astype(np.int64)
            dst_nodes = coo.col.astype(np.int64)
            num_nodes = adjacency.shape[0]
        else:
            adjacency = np.asarray(adjacency)
            src_nodes, dst_nodes = np.nonzero(adjacency)
            src_nodes = src_nodes.astype(np.int64)
            dst_nodes = dst_nodes.astype(np.int64)
            num_nodes = adjacency.shape[0]
        features = np.asarray(graph_data['features'], dtype=np.float32)

    g = dgl.graph((src_nodes, dst_nodes), num_nodes=int(num_nodes))
    g = dgl.add_self_loop(g)

    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    if features.shape[0] != g.num_nodes():
        raise ValueError(
            f"Feature count does not match node count: protein_id={protein_id}, features={features.shape[0]}, nodes={g.num_nodes()}"
        )

    coords = get_calpha_coordinates(protein_id)
    node_count = features.shape[0]
    coord_count = len(coords)
    STRICT_COORD = os.getenv('PPI_STRICT_COORD_MATCH', '0') == '1'
    if coord_count != node_count:
        msg = f"len(coords)={coord_count} != nodes={node_count}"
        if STRICT_COORD:
            _append_fail_pdb(protein_id, msg + "[strict]")
            raise ValueError(f"Coordinate count does not match node count: {protein_id} ({msg})")
        import hashlib
        if coord_count == 0:
            _append_fail_pdb(protein_id, msg + "[empty]")
            raise ValueError(f"No coordinates: {protein_id}")
        seed_bytes = hashlib.md5(protein_id.encode('utf-8')).digest()
        seed_int = int.from_bytes(seed_bytes[:8], 'little') & 0xffffffff
        rng = np.random.default_rng(seed_int)
        if coord_count > node_count:
            select_idx = np.sort(rng.choice(coord_count, size=node_count, replace=False))
            coords = coords[select_idx]
        else:
            repeat_times = node_count // coord_count
            remainder = node_count % coord_count
            coords = np.repeat(coords, repeat_times, axis=0)
            if remainder > 0:
                coords = np.vstack([coords, coords[:remainder]])
        if len(coords) != node_count:
            _append_fail_pdb(protein_id, msg + "[align-fail]")
            raise ValueError(f"Alignment failed: {protein_id} ({msg})")
        if DEBUG_DATA and not QUIET:
            print(f"[coords-align] {protein_id}: {msg} -> aligned {len(coords)}")

    centroid = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    dmin, dmax = distances.min(), distances.max()
    if dmax - dmin < 1e-12:
        time_steps = np.zeros((coords.shape[0], 1), dtype=np.float32)
    else:
        distances = (distances - dmin) / (dmax - dmin + 1e-8)
        time_steps = distances.reshape(-1, 1).astype(np.float32)

    features = np.hstack([features, time_steps])
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    g.ndata['fea'] = torch.from_numpy(features).float()

    if not disable_cache_in_worker:
        _cache_graph(protein_id, g)
    return g

def setup_seed(seed=867482):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getData_GCN(protein1: str, protein2: str) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
    g1 = get_protein_graph(protein1)
    g2 = get_protein_graph(protein2)
    return g1, g2

class CollateGCN:
    def __init__(self, label: str = "", every: int = 200, quiet: Optional[bool] = None):
        self.label = label
        self.every = every
        self.quiet = quiet if quiet is not None else QUIET
        self.processed = 0
        self.invalid = 0

    def __call__(self, samples: List[Tuple[str, str, int]]):
        graphs1, graphs2, labels = [], [], []
        local_invalid = 0
        for p1, p2, y in samples:
            try:
                g1, g2 = getData_GCN(p1, p2)
                graphs1.append(g1)
                graphs2.append(g2)
                labels.append(y)
            except Exception as e:
                local_invalid += 1
                append_fail_cache(f"{p1}|{p2}", f"graph-build: {e}", label=self.label or "collate")
        self.processed += len(samples)
        self.invalid += local_invalid
        if self.processed % max(1, self.every) == 0 and not self.quiet:
            print(f"📦 [{self.label}] collate progress: processed {self.processed} samples, invalid this batch {local_invalid}, total invalid {self.invalid}")
        if graphs1 and graphs2:
            return dgl.batch(graphs1), dgl.batch(graphs2), torch.tensor(labels)
        else:
            if not self.quiet:
                print(f"⚠️ [{self.label}] No valid samples in batch (invalid this batch {local_invalid}, total invalid {self.invalid})")
            return None

def collate_GCN(samples: List[Tuple[str, str, int]], quiet: Optional[bool] = None):
    if quiet is None:
        quiet = QUIET
    graphs1, graphs2, labels = [], [], []
    invalid_count = 0
    for p1, p2, label in samples:
        try:
            g1, g2 = getData_GCN(p1, p2)
            graphs1.append(g1)
            graphs2.append(g2)
            labels.append(label)
        except Exception as e:
            invalid_count += 1
            append_fail_cache(f"{p1}|{p2}", f"graph-build: {e}", label="collate")
    if graphs1 and graphs2:
        return dgl.batch(graphs1), dgl.batch(graphs2), torch.tensor(labels)
    if not quiet:
        print(f"⚠️ [collate] No valid samples in batch ({invalid_count} invalid entries in total)")
    return None

def make_collate_GCN(label: str = "", every: int = 200, quiet: Optional[bool] = None) -> CollateGCN:
    return CollateGCN(label=label, every=every, quiet=quiet)
