import os
import sys
import numpy as np
import h5py
import threading
import time
import signal
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdjacencyMatrixGenerator:
    def __init__(self):
        # Configuration paths
        self.base_dir = os.path.join(
            "your_basedir"
        )
        self.pdb_dir = os.path.join(self.base_dir, "pdb")
        self.rsa_dir = os.path.join(self.base_dir, "freesasa")
        self.dssp_dir = os.path.join(self.base_dir, "DSSP")
        self.output_dir = os.path.join(self.base_dir, "npz")
        self.fail_file = os.path.join(self.base_dir, "fail")
        self.progress_file = os.path.join(self.base_dir, "aprogress.json")

        # ESM data paths
        self.esm_mean_emb_path = os.path.join(self.base_dir, "esm_mean_embeddings.npy")
        self.esm_ids_path = os.path.join(self.base_dir, "esm_ids.npy")
        self.esm_if1_h5_path = os.path.join(self.base_dir, "esmif1.h5")

        # Parameter configuration
        self.distance_cutoff = 8.0
        self.rsa_interaction_threshold = 0.25  # RSA threshold, used to determine interaction propensity
        self.max_workers = min(12, os.cpu_count())  # Adjust the number of threads based on CPU cores

        # Theoretical maximum SASA values
        self.max_sasa_values = {
            'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
            'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
            'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
            'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.fail_file, 'w', encoding='utf-8') as f:
            f.write(f"# Batch processing failure records - {datetime.now()}\n")
            f.write(f"# RSA interaction threshold: {self.rsa_interaction_threshold}\n")
        self.load_esm_data()
        self.lock = threading.Lock()
        self.failed_ids = []
        self.processed_ids = []
        self.start_time = None
        self.should_stop = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, gracefully shutting down...")
        self.should_stop = True

    def load_esm_data(self):
        try:
            self.esm_ids = np.load(self.esm_ids_path)
            self.esm_means = np.load(self.esm_mean_emb_path)
            logger.info(f"Successfully loaded ESM data: IDs shape={self.esm_ids.shape}, Means shape={self.esm_means.shape}")
        except Exception as e:
            logger.warning(f"ESM data loading failed: {e}")
            self.esm_ids = None
            self.esm_means = None

    def get_esm_features(self, pdb_id):
        if self.esm_ids is None or self.esm_means is None:
            raise FileNotFoundError(f"{pdb_id}: ESM data file does not exist, unable to obtain sequence features")

        try:
            esm_ids_str = self.esm_ids.astype(str)

            # Method 1: Direct match
            indices = np.where(esm_ids_str == pdb_id)[0]
            if len(indices) == 1:
                logger.info(f"{pdb_id}: Direct match found ESM data")
                return self.esm_means[indices[0]]

            # Method 2: Match various UniProt formats
            pattern_matches = []
            for i, esm_id in enumerate(esm_ids_str):
                # Support multiple prefixes: sp|, tr|, etc.
                if (esm_id.startswith(f"sp|{pdb_id}|") or
                    esm_id.startswith(f"tr|{pdb_id}|") or
                    esm_id.startswith(f"sw|{pdb_id}|") or
                    esm_id.startswith(f"gb|{pdb_id}|") or
                    esm_id.startswith(f"ref|{pdb_id}|") or
                    esm_id.startswith(f"emb|{pdb_id}|") or
                    esm_id.startswith(f"dbj|{pdb_id}|") or
                    esm_id.startswith(f"pir|{pdb_id}|") or
                    esm_id.startswith(f"prf|{pdb_id}|") or
                    esm_id.startswith(f"pdb|{pdb_id}|")):
                    pattern_matches.append(i)

            if len(pattern_matches) == 1:
                logger.info(f"{pdb_id}: Found ESM data via pattern matching: {esm_ids_str[pattern_matches[0]]}")
                return self.esm_means[pattern_matches[0]]
            elif len(pattern_matches) > 1:
                logger.info(f"{pdb_id}: Found multiple matches, using the first: {esm_ids_str[pattern_matches[0]]}")
                return self.esm_means[pattern_matches[0]]

            # Method 3: Flexible matching - find entries containing the target ID
            flexible_matches = []
            for i, esm_id in enumerate(esm_ids_str):
                if pdb_id in esm_id:
                    parts = esm_id.split('|')
                    if pdb_id in parts:
                        flexible_matches.append(i)

            if len(flexible_matches) == 1:
                logger.info(f"{pdb_id}: Found ESM data via flexible matching: {esm_ids_str[flexible_matches[0]]}")
                return self.esm_means[flexible_matches[0]]
            elif len(flexible_matches) > 1:
                logger.info(f"{pdb_id}: Flexible matching found multiple results, using the first: {esm_ids_str[flexible_matches[0]]}")
                return self.esm_means[flexible_matches[0]]
            logger.warning(f"{pdb_id}: No match found. First 10 ID examples from ESM data:")
            for i, esm_id in enumerate(esm_ids_str[:100]):
                logger.warning(f"  Example {i}: {esm_id}")
            raise KeyError(f"{pdb_id}: No matching sequence features found in ESM data")
        except Exception as e:
            if isinstance(e, (KeyError, FileNotFoundError)):
                raise e
            else:
                raise RuntimeError(f"{pdb_id}: Exception while retrieving ESM features - {e}")

    def calculate_interaction_propensity(self, rsa_values):
        interaction_propensity = []
        high_rsa_count = 0

        for rsa_val in rsa_values:
            # Residues with RSA > 0.25 are more likely to participate in protein interactions
            if rsa_val > self.rsa_interaction_threshold:
                propensity = 1.0  # High interaction propensity
                high_rsa_count += 1
            else:
                propensity = 0.0  # Low interaction propensity

            interaction_propensity.append(propensity)
        # Compute interaction propensity statistics for the whole protein
        total_residues = len(rsa_values)
        interaction_ratio = high_rsa_count / total_residues if total_residues > 0 else 0.0
        return np.array(interaction_propensity), {
            'high_interaction_residues': high_rsa_count,
            'total_residues': total_residues,
            'interaction_ratio': interaction_ratio,
            'rsa_threshold': self.rsa_interaction_threshold
        }

    def analyze_surface_accessibility(self, rsa_values):
        rsa_array = np.array(rsa_values)
        return {
            'mean_rsa': np.mean(rsa_array),
            'std_rsa': np.std(rsa_array),
            'min_rsa': np.min(rsa_array),
            'max_rsa': np.max(rsa_array),
            'buried_residues': np.sum(rsa_array <= 0.1),  # Buried residues (RSA ≤ 0.1)
            'partially_exposed': np.sum((rsa_array > 0.1) & (rsa_array <= 0.25)),  # Partially exposed
            'interaction_prone': np.sum(rsa_array > 0.25),  # Interaction-prone residues (RSA > 0.25)
            'highly_exposed': np.sum(rsa_array > 0.5)  # Highly exposed (RSA > 0.5)
        }
    def process_single_protein(self, pdb_id):
        try:
            pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
            rsa_path = os.path.join(self.rsa_dir, f"{pdb_id}_complete.rsa")
            dssp_path = os.path.join(self.dssp_dir, f"{pdb_id}.dssp")
            missing_files = []
            if not os.path.exists(pdb_path):
                missing_files.append("PDB")
            if not os.path.exists(rsa_path):
                missing_files.append("RSA")
            if not os.path.exists(dssp_path):
                missing_files.append("DSSP")
            if missing_files:
                raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

            # 1. Read PDB file
            residues, coords = self.read_pdb(pdb_path)
            if len(residues) == 0:
                raise ValueError("No CA atoms found")
            M = len(residues)

            # 2. Build distance matrix and adjacency matrix
            coords = np.array(coords)
            diff = coords[:, None, :] - coords[None, :, :]
            dist_mat = np.linalg.norm(diff, axis=-1)
            adjacency = (dist_mat < self.distance_cutoff).astype(np.int8)
            np.fill_diagonal(adjacency, 0)

            # 3. Read RSA data (strict mode)
            rsa_map = self.read_rsa(rsa_path, pdb_id)

            # 4. Read DSSP data (strict mode)
            dssp_map = self.read_dssp(dssp_path, pdb_id)

            # 5. Get ESM-2 features (strict mode)
            esm_seq_vec = self.get_esm_features(pdb_id)

            # 6. Get ESM-IF1 features (strict mode)
            esm_if1_feats = self.get_esm_if1_features(pdb_id, M)

            # 7. Validate data completeness
            self.validate_data_completeness(pdb_id, residues, rsa_map, dssp_map, esm_seq_vec, esm_if1_feats)

            # 8. Compute protein interaction propensity features
            rsa_values = [rsa_map.get((ch, num), 0.0) for ch, num, _ in residues]
            interaction_propensity, interaction_stats = self.calculate_interaction_propensity(rsa_values)
            surface_analysis = self.analyze_surface_accessibility(rsa_values)

            # 9. Build feature matrix
            features = self.build_features(residues, esm_seq_vec, esm_if1_feats, rsa_map, dssp_map,
                                           interaction_propensity)

            # 10. Save results
            output_path = os.path.join(self.output_dir, f"{pdb_id}_graph.npz")
            np.savez(
                output_path,
                adjacency=adjacency,
                features=features,
                residues=residues,
                pdb_id=pdb_id,
                cutoff=self.distance_cutoff,
                distance_matrix=dist_mat,
                interaction_propensity=interaction_propensity,
                interaction_stats=interaction_stats,
                surface_analysis=surface_analysis,
                rsa_values=np.array(rsa_values),
                rsa_threshold=self.rsa_interaction_threshold,
                esm2_available=True,
                esm_if1_available=True
            )

            logger.info(f"{pdb_id}: Analysis - "
                        f"Interaction-prone residues: {interaction_stats['high_interaction_residues']}/{M} "
                        f"({interaction_stats['interaction_ratio']:.1%}), "
                        f"Feature dimension: {features.shape[1]}")
            return True, f"Successfully processed {pdb_id}: {M} residues, {np.sum(adjacency)} contacts, " \
                         f"Interaction-prone residue ratio: {interaction_stats['interaction_ratio']:.1%}"
        except Exception as e:
            error_msg = f"{pdb_id}: {str(e)}"
            with self.lock:
                self.failed_ids.append(error_msg)
                with open(self.fail_file, 'a', encoding='utf-8') as f:
                    f.write(f"{error_msg}\n")
            return False, error_msg

    def read_pdb(self, pdb_path):
        residues = []
        coords = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    chain = line[21]
                    resnum = int(line[22:26])
                    resname = line[17:20].strip()
                    x, y, z = map(float, (line[30:38], line[38:46], line[46:54]))
                    residues.append((chain, resnum, resname))
                    coords.append((x, y, z))
        return residues, coords

    def read_rsa(self, rsa_path, pdb_id):
        if not os.path.exists(rsa_path):
            raise FileNotFoundError(f"{pdb_id}: RSA file does not exist: {rsa_path}")
        rsa_map = {}
        valid_lines = 0
        absolute_sasa_values = []
        rsa_values = []
        with open(rsa_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('RES') and not line.startswith('RES  Chain'):
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            ch = parts[1]
                            resname = parts[2]
                            num = int(parts[3])
                            absolute_sasa = float(parts[4])
                            max_val = self.max_sasa_values.get(resname, 200) 
                            rsa = absolute_sasa / max_val
                            rsa = min(rsa, 1.0)
                            rsa_map[(ch, num)] = rsa
                            valid_lines += 1
                            absolute_sasa_values.append(absolute_sasa)
                            rsa_values.append(rsa)
                        except (ValueError, IndexError):
                            continue
        if valid_lines == 0:
            raise ValueError(f"{pdb_id}: No valid data lines found in RSA file")
        logger.info(f"{pdb_id}: Successfully read {valid_lines} RSA values")
        logger.info(
            f"{pdb_id}: Absolute SASA range: {np.min(absolute_sasa_values):.1f} - {np.max(absolute_sasa_values):.1f} Å")
        logger.info(f"{pdb_id}: Relative RSA range: {np.min(rsa_values):.3f} - {np.max(rsa_values):.3f}")
        logger.info(f"{pdb_id}: Mean RSA: {np.mean(rsa_values):.3f}")
        return rsa_map

    def validate_data_completeness(self, pdb_id, residues, rsa_map, dssp_map, esm_seq_vec, esm_if1_feats):
        M = len(residues)
        missing_data = []
        if esm_seq_vec is None or esm_seq_vec.shape[0] != 1280:
            missing_data.append("ESM-2 sequence features")
        if esm_if1_feats is None or esm_if1_feats.shape != (M, 512):
            missing_data.append(
                f"ESM-IF1 structural features (expected {M}x512, got {esm_if1_feats.shape if esm_if1_feats is not None else 'None'})")
        rsa_missing = 0
        for ch, num, _ in residues:
            if (ch, num) not in rsa_map:
                rsa_missing += 1
        if rsa_missing > M * 0.1:
            missing_data.append(f"RSA data (missing {rsa_missing}/{M} residues)")
        dssp_missing = 0
        for ch, num, _ in residues:
            if (ch, num) not in dssp_map:
                dssp_missing += 1
        if dssp_missing > M * 0.1: 
            missing_data.append(f"DSSP data (missing {dssp_missing}/{M} residues)")
        if missing_data:
            raise ValueError(f"{pdb_id}: Critical data is missing or incomplete: {', '.join(missing_data)}")
        logger.info(f"{pdb_id}: Data completeness validation passed")

    def read_dssp(self, dssp_path, pdb_id):
        if not os.path.exists(dssp_path):
            raise FileNotFoundError(f"{pdb_id}: DSSP file does not exist: {dssp_path}")
        dssp_map = {}
        valid_lines = 0
        with open(dssp_path) as f:
            found_header = False
            for L in f:
                if L.startswith("  #  RESIDUE"):
                    found_header = True
                    break
            if not found_header:
                raise ValueError(f"{pdb_id}: Invalid DSSP format, data header not found")
            for L in f:
                if len(L) < 17:
                    continue
                ch = L[11]
                try:
                    num = int(L[5:10])
                except ValueError:
                    continue
                ss = L[16]
                dssp_map[(ch, num)] = ss
                valid_lines += 1
        if valid_lines == 0:
            raise ValueError(f"{pdb_id}: No valid data lines found in DSSP file")
        logger.info(f"{pdb_id}: Successfully read {valid_lines} DSSP values")
        return dssp_map

    def get_esm_if1_features(self, pdb_id, M):
        try:
            with h5py.File(self.esm_if1_h5_path, 'r') as h5:
                if pdb_id in h5['proteins']:
                    grp = h5['proteins'][pdb_id]
                    logger.info(f"{pdb_id}: Protein data found directly in HDF5")
                else:
                    matching_keys = [k for k in h5['proteins'].keys() if pdb_id in k]
                    if matching_keys:
                        grp = h5['proteins'][matching_keys[0]]
                        logger.info(f"{pdb_id}: HDF5 data found via pattern matching: {matching_keys[0]}")
                    else:
                        raise KeyError(f"{pdb_id}: Protein data not found in ESM-IF1 HDF5 file")
                available_chains = list(grp.keys())
                if not available_chains:
                    raise ValueError(f"{pdb_id}: No available chain data in HDF5")
                chain_grp = grp[available_chains[0]]
                if 'features' not in chain_grp:
                    raise KeyError(f"{pdb_id}: No features data found in chain {available_chains[0]}")
                features = chain_grp['features'][:]
                logger.info(f"{pdb_id}: Loaded ESM-IF1 features, shape: {features.shape}")
                if features.shape[0] != M:
                    if features.shape[0] > M:
                        logger.warning(f"{pdb_id}: ESM-IF1 feature length ({features.shape[0]}) > CA atom count ({M}), truncating to {M}")
                        features = features[:M]
                    else:
                        raise ValueError(f"{pdb_id}: ESM-IF1 feature length ({features.shape[0]}) < CA atom count ({M}), data is incomplete")
                return features
        except Exception as e:
            if isinstance(e, (KeyError, ValueError, FileNotFoundError)):
                raise e
            else:
                raise RuntimeError(f"{pdb_id}: Exception while retrieving ESM-IF1 features - {e}")

    def build_features(self, residues, esm_seq_vec, esm_if1_feats, rsa_map, dssp_map, interaction_propensity):
        ss_types = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'P', ' ']
        ss_oh = {ss: np.eye(len(ss_types))[i] for i, ss in enumerate(ss_types)}
        features = []
        for i, (ch, num, _) in enumerate(residues):
            seq_vec = esm_seq_vec  # 1280D - ESM-2 sequence features
            str_vec = esm_if1_feats[i]  # 512D  - ESM-IF1 structural features
            rsa_val = rsa_map.get((ch, num), 0.0)  # 1D    - RSA solvent accessibility
            ss_char = dssp_map.get((ch, num), ' ')
            interaction_prop = interaction_propensity[i]  # 1D - protein interaction propensity
            if ss_char not in ss_oh:
                ss_char = ' '
            ss_vec = ss_oh[ss_char]  # 9D    - DSSP secondary structure
            # Concatenate all features: 1280 + 512 + 1 + 9 + 1 = 1803D
            feat = np.concatenate([seq_vec, str_vec, [rsa_val], ss_vec, [interaction_prop]])
            features.append(feat)
        return np.vstack(features)

    def save_progress(self, total, completed, success_count, failed_count):
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'total': total,
            'completed': completed,
            'success_count': success_count,
            'failed_count': failed_count,
            'processed_ids': self.processed_ids,
            'failed_ids': self.failed_ids,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'rsa_threshold': self.rsa_interaction_threshold  # Record RSA threshold
        }
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def load_progress(self):
        if not os.path.exists(self.progress_file):
            return None
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return None

    def get_system_info(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(self.base_dir)
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024 ** 3),
                'disk_free_gb': disk.free / (1024 ** 3)
            }
        except Exception:
            return {}

    def run_batch_processing(self, protein_ids):
        self.start_time = datetime.now()
        progress_data = self.load_progress()
        if progress_data:
            logger.info(f"Previous progress record found, already processed: {progress_data.get('completed', 0)}")
            self.processed_ids = progress_data.get('processed_ids', [])
            self.failed_ids = progress_data.get('failed_ids', [])
            remaining_ids = [pid for pid in protein_ids if pid not in self.processed_ids]
            logger.info(f"Remaining proteins to process: {len(remaining_ids)}")
        else:
            remaining_ids = protein_ids
        logger.info(f"Starting batch processing for {len(remaining_ids)} proteins")
        logger.info(f"RSA interaction threshold: {self.rsa_interaction_threshold}")
        logger.info(f"Using {self.max_workers} threads")
        sys_info = self.get_system_info()
        if sys_info:
            logger.info(f"System status - CPU: {sys_info.get('cpu_percent', 0):.1f}%, "
                        f"Memory: {sys_info.get('memory_percent', 0):.1f}%, "
                        f"Available memory: {sys_info.get('memory_available_gb', 0):.1f}GB")
        success_count = len(
            [pid for pid in self.processed_ids if pid not in [f.split(':')[0] for f in self.failed_ids]])
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {executor.submit(self.process_single_protein, pid): pid
                            for pid in remaining_ids}
            initial_completed = len(self.processed_ids)
            with tqdm(total=len(protein_ids), initial=initial_completed, desc="Processing progress",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
                for future in as_completed(future_to_id):
                    if self.should_stop:
                        logger.info("Stop signal received, saving progress...")
                        break
                    protein_id = future_to_id[future]
                    try:
                        success, message = future.result()
                        with self.lock:
                            self.processed_ids.append(protein_id)
                            if success:
                                success_count += 1
                            else:
                                logger.error(f"Processing failed: {message}")
                            pbar.set_postfix({
                                "success": success_count,
                                "failed": len(self.failed_ids),
                                "memory": f"{psutil.virtual_memory().percent:.1f}%"
                            })
                            if len(self.processed_ids) % 10 == 0:
                                self.save_progress(len(protein_ids), len(self.processed_ids),
                                                   success_count, len(self.failed_ids))
                    except Exception as e:
                        error_msg = f"{protein_id}: Unknown error - {str(e)}"
                        logger.error(error_msg)
                        with self.lock:
                            self.failed_ids.append(error_msg)
                            self.processed_ids.append(protein_id)
                    pbar.update(1)
        self.save_progress(len(protein_ids), len(self.processed_ids),
                           success_count, len(self.failed_ids))
        elapsed_time = datetime.now() - self.start_time
        logger.info(f"Batch processing completed! Time elapsed: {elapsed_time}")
        logger.info(f"Successfully processed: {success_count}/{len(protein_ids)}")
        logger.info(f"Failed count: {len(self.failed_ids)}")
        logger.info(f"Failure records saved to: {self.fail_file}")
        logger.info(f"Progress record saved to: {self.progress_file}")
        return success_count, len(self.failed_ids)

def main():
    try:
        protein_list_file = os.path.join(
            "your_basedir",
            "your.txt"
        )
        if not os.path.exists(protein_list_file):
            logger.error(f"Protein list file does not exist: {protein_list_file}")
            return
        with open(protein_list_file, 'r', encoding='utf-8') as f:
            protein_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(protein_ids)} protein IDs from {protein_list_file}")
        generator = AdjacencyMatrixGenerator()
        success_count, fail_count = generator.run_batch_processing(protein_ids)
        print(f"\n{'=' * 70}")
        print(f"🧬 Batch processing completed!")
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {fail_count}")
        print(f"📊 Feature dimension: 1803D")
        print(f"  • ESM-2 sequence features: 1280D")
        print(f"  • ESM-IF1 structural features: 512D")
        print(f"  • RSA solvent accessibility: 1D")
        print(f"  • DSSP secondary structure: 9D")
        print(f"  • Protein interaction propensity: 1D (RSA > {generator.rsa_interaction_threshold})")
        print(f"📁 Results saved in: {generator.output_dir}")
        print(f"📋 Failure records: {generator.fail_file}")
        print(f"{'=' * 70}")
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()