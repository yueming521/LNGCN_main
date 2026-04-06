import os
import sys
import pickle
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import dgl
from tqdm import tqdm
import importlib.util

def import_model_from_file(filepath: str, name: str):
	spec = importlib.util.spec_from_file_location("_ext_module", filepath)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module: {filepath}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	if not hasattr(module, name):
		raise AttributeError(f"Object not found in {filepath}: {name}")
	return getattr(module, name)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MYGCN_PATH = os.path.join(BASE_DIR, "main", "main_model.py")
DATA_MODULE_PATH = os.path.join(BASE_DIR, "calibration_and_predictor", "data_predictor.py")

MyGCN = import_model_from_file(MYGCN_PATH, "MyGCN")
getData_GCN = import_model_from_file(DATA_MODULE_PATH, "getData_GCN")
check_protein_exists = import_model_from_file(DATA_MODULE_PATH, "check_protein_exists")

class PredictorConfig:
	def __init__(self, candidate_base_dirs: Optional[List[str]] = None):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.batch_size = 64

		# 10 binary models (5 folds x AUPRC/AUROC)
		root_bh = "LNGCN_main/results/balance_human"
		self.model_paths: List[str] = [
			os.path.join(root_bh, f"fold{fold}_best_auprc.pt")
			for fold in range(1, 6)
		] + [
			os.path.join(root_bh, f"fold{fold}_best_auroc.pt")
			for fold in range(1, 6)
		]

		default_candidate_dirs = [
			"LNGCN_main/results/calibrate/evaluate_summary",
		]

		extra_env = os.environ.get("CALIBRATOR_BASE_DIRS", "").strip()
		if extra_env:
			default_candidate_dirs.extend([p.strip() for p in extra_env.split(",") if p.strip()])

		candidate_base_dirs = candidate_base_dirs or default_candidate_dirs

		self.calibrator_methods = ("platt", "beta", "boosting", "isotonic")
		self.calibrator_metrics = ("auprc", "auroc")

		best_paths: List[str] = []
		best_count = -1
		for base_calib in candidate_base_dirs:
			paths: List[str] = []
			for metric in self.calibrator_metrics:
				calib_dir = os.path.join(base_calib, f"calibrated_models_{metric}")
				for method in self.calibrator_methods:
					for fold in range(1, 6):
						paths.append(os.path.join(calib_dir, f"calibrator_{method}_fold{fold}.pkl"))
			count = sum(os.path.exists(p) for p in paths)
			if count > best_count:
				best_paths = paths
				best_count = count

		self.calibrator_paths = best_paths
		if best_count <= 0:
			print("⚠️ No calibrators found in candidate dirs; will try default paths (may be empty)")

class EnsembleBinaryPredictor:
	def __init__(self, config: PredictorConfig):
		self.config = config
		self.device = config.device
		self.models: List[nn.Module] = []
		self.in_dim: Optional[int] = None

		print(f"🔧 Loading {len(config.model_paths)} binary PT models...")
		for i, path in enumerate(config.model_paths, 1):
			model = self._load_single_model(path)
			self.models.append(model)
			print(f"  ✅ Model {i}/{len(config.model_paths)}: {os.path.basename(path)}")

	def _load_single_model(self, model_path: str) -> nn.Module:
		if not os.path.exists(model_path):
			raise FileNotFoundError(f"Model file not found: {model_path}")

		state = torch.load(model_path, map_location=self.device)
		state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

		key = "cfc_preprocess.cells.0.backbone.0.weight"
		if key in state_dict:
			in_dim_minus_1 = state_dict[key].shape[1]
		elif f"module.{key}" in state_dict:
			in_dim_minus_1 = state_dict[f"module.{key}"].shape[1]
		else:
			raise ValueError(f"Key not found in checkpoint {key}: {model_path}")

		in_dim = in_dim_minus_1 + 1
		if self.in_dim is None:
			self.in_dim = in_dim
		else:
			if self.in_dim != in_dim:
				print(f"⚠️ Inferred in_dim differs across models; using {self.in_dim}")
				in_dim = self.in_dim

		model = MyGCN(in_dim=in_dim)

		new_state = {}
		for k, v in state_dict.items():
			if k.startswith("module."):
				new_state[k[7:]] = v
			else:
				new_state[k] = v

		model.load_state_dict(new_state, strict=False)
		model.to(self.device)
		model.eval()
		return model

	def predict_logits_batch(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
		if not pairs:
			return np.zeros((0,), dtype=np.float32)

		g1_list, g2_list, valid_pairs = [], [], []
		for p1, p2 in pairs:
			try:
				if not (check_protein_exists(p1) and check_protein_exists(p2)):
					raise ValueError("protein not found")
				g1, g2 = getData_GCN(p1, p2)
				g1_list.append(g1)
				g2_list.append(g2)
				valid_pairs.append((p1, p2))
			except Exception as e:
				print(f"⚠️ Skipped {p1}-{p2}: {e}")

		if not valid_pairs:
			return np.zeros((0,), dtype=np.float32)

		bg1 = dgl.batch(g1_list).to(self.device)
		bg2 = dgl.batch(g2_list).to(self.device)
		fea1 = bg1.ndata["fea"].to(self.device, dtype=torch.float32)
		fea2 = bg2.ndata["fea"].to(self.device, dtype=torch.float32)

		all_logits = []
		with torch.no_grad():
			for model in self.models:
				out = model(bg1, bg2, fea1, fea2)  # (N, 2)
				logits = out[:, 1].cpu().numpy()
				all_logits.append(logits)

		mean_logits = np.mean(all_logits, axis=0)
		return mean_logits

	def predict_all_logits_batch(self, pairs: List[Tuple[str, str]]) -> List[np.ndarray]:
		if not pairs:
			return [np.zeros((0,), dtype=np.float32) for _ in self.models]
		g1_list, g2_list, valid_pairs = [], [], []
		for p1, p2 in pairs:
			try:
				if not (check_protein_exists(p1) and check_protein_exists(p2)):
					raise ValueError("protein not found")
				g1, g2 = getData_GCN(p1, p2)
				g1_list.append(g1)
				g2_list.append(g2)
				valid_pairs.append((p1, p2))
			except Exception as e:
				print(f"⚠️ Skipped {p1}-{p2}: {e}")

		if not valid_pairs:
			return [np.zeros((0,), dtype=np.float32) for _ in self.models]

		bg1 = dgl.batch(g1_list).to(self.device)
		bg2 = dgl.batch(g2_list).to(self.device)
		fea1 = bg1.ndata["fea"].to(self.device, dtype=torch.float32)
		fea2 = bg2.ndata["fea"].to(self.device, dtype=torch.float32)

		all_logits = []
		with torch.no_grad():
			for model in self.models:
				out = model(bg1, bg2, fea1, fea2) 
				logits = out[:, 1].cpu().numpy()
				all_logits.append(logits)

		return all_logits


def _sigmoid(x: np.ndarray) -> np.ndarray:
	return 1 / (1 + np.exp(-np.clip(x, -500, 500))) 


class DictBasedCalibrator:
	"""Supported methods: platt/beta/temperature/isotonic/boosting
	- platt: p = sigmoid(A*logit + B)
	- beta:  p = sigmoid(logit); p' = sigmoid(a*log(p)+b*log(1-p)+c)
	- temperature: p = sigmoid(logit/temperature)
	- isotonic: p = sigmoid(logit) mapped to interpolated y=interp(p; x,y)
	- boosting: predict with a trained GradientBoostingClassifier model
	"""

	def __init__(self, payload: dict):
		self.payload = payload
		self.method = str(payload.get("method", "")).lower()
		self.params = payload.get("params", {}) or {}
		self.meta = payload.get("meta", {}) or {}
		self.version = payload.get("version", 1)

	def predict_proba(self, logits: np.ndarray) -> np.ndarray:
		z = np.asarray(logits, dtype=float).reshape(-1)
		m = self.method
		
		try:
			if m == "platt":
				A = float(self.params.get("A", 1.0))
				B = float(self.params.get("B", 0.0))
				return _sigmoid(A * z + B)
			
			elif m == "beta":
				p = _sigmoid(z)
				eps = 1e-15
				p = np.clip(p, eps, 1 - eps)
				a = float(self.params.get("a", 0.0))
				b = float(self.params.get("b", 0.0))
				c = float(self.params.get("c", 0.0))
				return _sigmoid(a * np.log(p) + b * np.log(1.0 - p) + c)
			
			elif m == "temperature":
				T = float(self.params.get("temperature", 1.0))
				T = max(T, 1e-6)
				return _sigmoid(z / T)
			
			elif m == "isotonic":
				x = np.asarray(self.params.get("x", []), dtype=float)
				y = np.asarray(self.params.get("y", []), dtype=float)
				if x.size == 0 or y.size == 0 or x.size != y.size:
					print("⚠️ Invalid isotonic params; falling back to sigmoid")
					return _sigmoid(z)
				p = _sigmoid(z)
				# Linear interpolation, out-of-range uses endpoint values (clip equivalent)
				return np.interp(p, x, y, left=y[0], right=y[-1]).astype(float)
			
			elif m == "boosting":
				# Boosting calibrator: predict with full model object
				if 'model' in self.payload:
					from sklearn.ensemble import GradientBoostingClassifier
					model = self.payload['model']
					if not isinstance(model, GradientBoostingClassifier):
						print(f"⚠️ Boosting model type mismatch: {type(model)}")
						return _sigmoid(z)
					
					try:
						probs = model.predict_proba(z.reshape(-1, 1))[:, 1]
						return probs
					except Exception as e:
						print(f"⚠️ Boosting prediction failed: {e}")
						return _sigmoid(z)
				else:
					print("⚠️ Boosting calibrator missing full model object; cannot predict accurately")
					return _sigmoid(z)
			
			else:
				print(f"⚠️ Unknown calibration method '{m}', falling back to sigmoid")
				return _sigmoid(z)
				
		except Exception as e:
			print(f"⚠️ Calibration method '{m}' failed: {e}, falling back to sigmoid")
			return _sigmoid(z)
	
	def __repr__(self):
		return f"DictBasedCalibrator(method={self.method}, version={self.version})"

class PlattCalibratorEnsemble:
	def __init__(self, calibrator_paths: List[str]):
		self.calibrators: List[DictBasedCalibrator] = []
		self.names: List[str] = []

		print(f"🔧 Loading calibrators, {len(calibrator_paths)} candidates...")
		for i, path in enumerate(calibrator_paths, 1):
			if not os.path.exists(path):
				print(f"  ⚠️ Calibrator not found, skipping: {path}")
				continue
			try:
				with open(path, "rb") as f:
					data = pickle.load(f)
				
				if isinstance(data, dict) and 'method' in data and 'params' in data:
					calib = DictBasedCalibrator(data)
					if 'calibrated_models_auprc' in path:
						metric = 'auprc'
					elif 'calibrated_models_auroc' in path:
						metric = 'auroc'
					elif 'calibrated_models_f1' in path:
						metric = 'f1'
					else:
						metric = 'unknown'
					name = f"{os.path.basename(path).replace('.pkl', '')}_{metric}"
					self.calibrators.append(calib)
					self.names.append(name)
					method_info = f"{calib.method}" + (f" (v{calib.version})" if calib.version > 1 else "")
					print(f"  ✅ Calibrator {i}: {name} [{method_info}]")
				else:
					print(f"  ⚠️ Unsupported calibrator format (dict required), skipping: {path}")
					continue
					
			except Exception as e:
				print(f"  ❌ Load failed {path}: {e}")
				import traceback
				traceback.print_exc()

		if not self.calibrators:
			raise RuntimeError("No calibrators loaded. Ensure new-format .pkl files are generated.")

	def calibrate_all(self, logits: np.ndarray) -> Dict[str, np.ndarray]:
		results: Dict[str, np.ndarray] = {}
		for calib, name in zip(self.calibrators, self.names):
			try:
				probs = calib.predict_proba(logits)
				results[name] = np.asarray(probs, dtype=float)
			except Exception as e:
				print(f"⚠️ Calibrator {name} failed; falling back to sigmoid: {e}")
				import traceback
				traceback.print_exc()
				probs = _sigmoid(logits)
				results[name] = probs.astype(float)
		return results

	def aggregate_predictions(self, all_probs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
		"""Hierarchical weighted averaging.
		
		Input is per-model-per-calibrator probability dict all_probs:
		- key like: calibrator_platt_fold1_auprc, calibrator_beta_fold3_auroc, ...
		- value: probabilities for a batch, shape = (N,)
		
		Steps:
		1) Within fold: average 4 methods (platt/beta/boosting/isotonic) for each (fold, metric)
		   to get P_fold_k^AUPRC and P_fold_k^AUROC.
		2) Across metrics: average AUPRC and AUROC for each fold to get P_fold_k.
		3) Across folds: average 5 folds to get final ensemble_mean.
		
		"""
		if not all_probs:
			return {}

		# Parse all_probs into (fold, metric, method)
		# metric in {auprc, auroc}, method in {platt, beta, boosting, isotonic}
		methods = ["platt", "beta", "boosting", "isotonic"]
		metrics = ["auprc", "auroc"]
		fold_method_metric: Dict[Tuple[int, str, str], List[np.ndarray]] = {}

		for name, probs in all_probs.items():
			name_lower = name.lower()
			fold = None
			for k in range(1, 6):
				if f"fold{k}" in name_lower:
					fold = k
					break
			if fold is None:
				continue
			metric = None
			for m in metrics:
				if m in name_lower:
					metric = m
					break
			if metric is None:
				continue
			method = None
			for md in methods:
				if md in name_lower:
					method = md
					break
			if method is None:
				continue
			key = (fold, metric, method)
			fold_method_metric.setdefault(key, []).append(np.asarray(probs, dtype=float))

		if not fold_method_metric:
			print("⚠️ No (fold, metric, method) parsed from calibrations; returning empty aggregation")
			return {}

		# Level 1: within fold (avg methods per fold & metric)
		# Get P_fold_k^AUPRC / P_fold_k^AUROC
		fold_metric_probs: Dict[Tuple[int, str], np.ndarray] = {}
		for (fold, metric, _method), prob_list in fold_method_metric.items():
			key_fm = (fold, metric)
			stack = np.stack(prob_list, axis=0)
			mean_probs = np.mean(stack, axis=0)
			if key_fm in fold_metric_probs:
				fold_metric_probs[key_fm] = (fold_metric_probs[key_fm] + mean_probs) / 2.0
			else:
				fold_metric_probs[key_fm] = mean_probs

		# Level 2: average AUPRC and AUROC within fold to get P_fold_k
		fold_final: Dict[int, np.ndarray] = {}
		for fold in range(1, 6):
			arr = []
			for metric in metrics:
				key_fm = (fold, metric)
				if key_fm in fold_metric_probs:
					arr.append(fold_metric_probs[key_fm])
			if not arr:
				continue
			stack = np.stack(arr, axis=0)
			fold_final[fold] = np.mean(stack, axis=0)

		if not fold_final:
			print("⚠️ All folds missing AUPRC/AUROC info; returning empty aggregation")
			return {}

		# Level 3: bagging across folds to get final ensemble_mean
		stack_folds = np.stack(list(fold_final.values()), axis=0)
		ensemble_mean = np.mean(stack_folds, axis=0)
		return {"ensemble_mean": ensemble_mean}


class CalibratedPPIPredictor:
	def __init__(self, config: Optional[PredictorConfig] = None):
		self.config = config or PredictorConfig()
		self.ensemble = EnsembleBinaryPredictor(self.config)
		self.calibrator_ens = PlattCalibratorEnsemble(self.config.calibrator_paths)

	def predict(
		self,
		pairs_file: str,
		output_file_per: str,
		output_file_aggregated: Optional[str] = None,
		output_file_simple_agg: Optional[str] = None,
	) -> Tuple[List[Tuple[str, str, Dict[str, float]]], List[Tuple[str, str, Dict[str, float]]], List[Tuple[str, str, Dict[str, float]]]]:
		"""Predict protein pairs in `pairs_file` in batches and save results.
		
		Strategy:
		- output_file_per: PT-PKL paired results (each PT uses its metric calibrator, 40 total)
		- output_file_aggregated: hierarchical aggregation (recommended)
		  * ensemble_mean: global average (default recommended)
		- output_file_simple_agg: simple average over all calibrators
		  * ensemble_mean: global average
		  * method_X: per-method averages (platt/beta/boosting/isotonic)
		  * metric_X: per-metric averages (auprc/auroc)
		"""
		pairs: List[Tuple[str, str]] = []
		with open(pairs_file, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith("#"):
					continue
				parts = line.split()
				if len(parts) >= 2:
					pairs.append((parts[0], parts[1]))

		print(f"📂 Loaded protein pairs: {len(pairs)}")

		results_per: List[Tuple[str, str, Dict[str, float]]] = []
		results_aggregated: List[Tuple[str, str, Dict[str, float]]] = []
		results_simple_agg: List[Tuple[str, str, Dict[str, float]]] = []
		bs = self.config.batch_size
		start = time.time()

		for i in tqdm(range(0, len(pairs), bs), desc="Predicting"):
			batch_pairs = pairs[i : i + bs]
			
			# Option 1: PT-PKL pairing (each PT uses its metric calibrator)
			all_logits = self.ensemble.predict_all_logits_batch(batch_pairs)
			calib_probs_per = {}
			for model_idx, logits in enumerate(all_logits):
				if logits.size == 0:
					continue
				if model_idx < 5:
					metric = 'auprc'
					fold = model_idx + 1
				else:
					metric = 'auroc'
					fold = model_idx - 4
				
				for name in self.calibrator_ens.names:
					if f'fold{fold}' in name and f'_{metric}' in name:
						calib_idx = self.calibrator_ens.names.index(name)
						calib = self.calibrator_ens.calibrators[calib_idx]
						probs = calib.predict_proba(logits)
						calib_probs_per[name] = probs
			
			# Smart aggregation: hierarchical average
			aggregated_probs = self.calibrator_ens.aggregate_predictions(calib_probs_per)

			simple_agg_probs = {}
			if calib_probs_per:
				all_values = np.array(list(calib_probs_per.values()))
				simple_ensemble_mean = np.mean(all_values, axis=0)
				simple_agg_probs['ensemble_mean'] = simple_ensemble_mean
				
				# Aggregate by method
				method_groups = {'platt': [], 'beta': [], 'boosting': [], 'isotonic': []}
				for name, probs in calib_probs_per.items():
					for method in method_groups.keys():
						if method in name.lower():
							method_groups[method].append(probs)
							break
				
				# Aggregate by metric
				metric_groups = {'auprc': [], 'auroc': []}
				for name, probs in calib_probs_per.items():
					if 'auprc' in name.lower():
						metric_groups['auprc'].append(probs)
					elif 'auroc' in name.lower():
						metric_groups['auroc'].append(probs)
				
				for method, prob_list in method_groups.items():
					if prob_list:
						simple_agg_probs[f'method_{method}'] = np.mean(prob_list, axis=0)
				
				for metric, prob_list in metric_groups.items():
					if prob_list:
						simple_agg_probs[f'metric_{metric}'] = np.mean(prob_list, axis=0)

			idx = 0
			for p1, p2 in batch_pairs:
				if not (check_protein_exists(p1) and check_protein_exists(p2)):
					results_per.append((p1, p2, {}))
					results_aggregated.append((p1, p2, {}))
					results_simple_agg.append((p1, p2, {}))
					continue
				
				# Option 1 results (PT-PKL pairing)
				results_per.append((p1, p2, {
					name: float(probs[idx]) for name, probs in calib_probs_per.items()
				}))
				
				# Smart aggregation results (recommended)
				results_aggregated.append((p1, p2, {
					name: float(probs[idx]) for name, probs in aggregated_probs.items()
				}))

				# Simple average aggregation results
				results_simple_agg.append((p1, p2, {
					name: float(probs[idx]) for name, probs in simple_agg_probs.items()
				}))
				
				idx += 1

		elapsed = time.time() - start
		valid_count = len([r for r in results_per if r[2]])  # count of valid results
		print(f"✅ Prediction done, total {len(results_per)} (valid {valid_count}), elapsed {elapsed:.2f}s")
		print("\n📊 Result notes:")
		print(f"  1. Option 1 (PT-PKL pairing): {output_file_per}")
		print("     - Contains 40 paired results (more rigorous)")
		if output_file_aggregated:
			print(f"  2. Hierarchical aggregation (recommended): {output_file_aggregated}")
			print("     - ensemble_mean: global average (default recommended)")
		if output_file_simple_agg:
			print(f"  3. Simple average aggregation: {output_file_simple_agg}")
			print("     - ensemble_mean: global average")
			print("     - method_X: per-method average (platt/beta/boosting/isotonic)")
			print("     - metric_X: per-metric average (auprc/auroc)")

		# Save to CSV
		self._save_results_csv(results_per, output_file_per)
		if output_file_aggregated and results_aggregated:
			self._save_results_csv(results_aggregated, output_file_aggregated)
			print(f"\n💡 Tip: use the 'ensemble_mean' column in {output_file_aggregated} as the final prediction")
		if output_file_simple_agg and results_simple_agg:
			self._save_results_csv(results_simple_agg, output_file_simple_agg)
		
		return results_per, results_aggregated, results_simple_agg

	def _save_results_csv(
		self,
		results: List[Tuple[str, str, Dict[str, float]]],
		output_file: str,
	) -> None:
		os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
		if not results:
			print("⚠️ No results to save")
			return

		import csv

		all_keys = set()
		for _, _, prob_dict in results:
			all_keys.update(prob_dict.keys())
		calib_names = sorted(all_keys)

		with open(output_file, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["Protein1", "Protein2"] + calib_names)
			for p1, p2, prob_dict in results:
				if not prob_dict:
					row = [p1, p2] + ["feature_missing"] * len(calib_names)
				else:
					row = [p1, p2] + [prob_dict.get(name, "") for name in calib_names]
				writer.writerow(row)


def main():
	# Update to your actual test data paths as needed
	import argparse
	parser = argparse.ArgumentParser(description="Calibrated PPI Predictor")
	parser.add_argument("--pairs", default="yours/prediction.txt", help="Protein pair file path")
	parser.add_argument("--out-per", default="yours/predictions_calibrated.csv", help="Option 1: PT-PKL paired output CSV path")
	parser.add_argument("--out-agg", default="yours/aggregated_per_predictions.csv", help="Hierarchical aggregation output CSV path (recommended)")
	parser.add_argument("--calib-dirs", default=None, help="Custom calibrator base dirs, comma-separated. Each should point to a fold_jiaozhun dir")
	args = parser.parse_args()


	pairs_file = args.pairs
	output_file_per = args.out_per
	output_file_agg = args.out_agg
	output_file_simple_agg = args.out_simple_agg

	candidate_dirs = None
	if args.calib_dirs:
		candidate_dirs = [p.strip() for p in args.calib_dirs.split(",") if p.strip()]

	print("🧬 Calibrated PPI probability predictor (MyGCN + multi-method calibration)")
	print(f"📁 Protein pair file: {pairs_file}")
	print(f"📁 Option 1 output: {output_file_per}")
	print(f"📁 Hierarchical aggregation output: {output_file_agg}")
	print(f"📁 Simple average aggregation output: {output_file_simple_agg}")

	if not os.path.exists(pairs_file):
		print(f"❌ Protein pair file not found: {pairs_file}")
		return

	config = PredictorConfig(candidate_base_dirs=candidate_dirs)
	predictor = CalibratedPPIPredictor(config)
	predictor.predict(pairs_file, output_file_per, output_file_agg, output_file_simple_agg)


if __name__ == "__main__":
	main()

