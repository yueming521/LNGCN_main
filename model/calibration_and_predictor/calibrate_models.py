import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import json
from tqdm import tqdm
import importlib.util
from typing import Optional, Tuple

def import_model_from_file(filepath, class_name):
    spec = importlib.util.spec_from_file_location("model_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MyGCN = import_model_from_file(os.path.join(BASE_DIR, "main", "main_model.py"), "MyGCN")
getData_GCN = import_model_from_file(os.path.join(current_dir, "data_calibrate.py"), "getData_GCN")
check_protein_exists = import_model_from_file(os.path.join(current_dir, "data_calibrate.py"), "check_protein_exists")

class PlattScaling:
    def __init__(self, C=1.0, class_weight='balanced', max_iter=1000):
        from sklearn.linear_model import LogisticRegression
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.model = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter)
        self._fit_meta = {}

    def fit(self, logits, labels):
        self.model.fit(logits.reshape(-1, 1), labels)
        labels_arr = np.asarray(labels)
        pos_ratio = float(np.mean(labels_arr)) if len(labels_arr) > 0 else 0.0
        self._fit_meta = {
            'n_samples': int(len(labels_arr)),
            'train_positive_ratio': pos_ratio,
            'fit_time': pd.Timestamp.now().isoformat()
        }

    def predict_proba(self, logits):
        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]

    def save(self, path, method: str = 'platt', model_type: str | None = None, fold: int | None = None):
        params = self.get_params() or {}
        data = {
            'method': method,
            'version': 1,
            'params': {k: float(v) for k, v in params.items()},
            'meta': {
                **self._fit_meta,
                'model_type': model_type,
                'fold': fold
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'params' in data:
            params = data['params']
            if 'A' in params and 'B' in params:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter)
                dummy_X = np.array([[0], [1]])
                dummy_y = np.array([0, 1])
                self.model.fit(dummy_X, dummy_y)
                self.model.coef_ = np.array([[params['A']]])
                self.model.intercept_ = np.array([params['B']])
                self._fit_meta = data.get('meta', {})
                print(f"Loaded Platt params: A={params['A']:.4f}, B={params['B']:.4f}")
        else:
            self.model = data
            print("Loaded legacy Platt model")
    
    @classmethod
    def from_dict(cls, data: dict):
        calibrator = cls()
        params = data['params']
        if 'A' in params and 'B' in params:
            from sklearn.linear_model import LogisticRegression
            calibrator.model = LogisticRegression()
            dummy_X = np.array([[0], [1]])
            dummy_y = np.array([0, 1])
            calibrator.model.fit(dummy_X, dummy_y)
            calibrator.model.coef_ = np.array([[params['A']]])
            calibrator.model.intercept_ = np.array([params['B']])
            calibrator._fit_meta = data.get('meta', {})
        return calibrator

    def get_params(self):
        if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            A = self.model.coef_[0][0] 
            B = self.model.intercept_[0]
            return {'A': A, 'B': B}
        else:
            return None


class BetaCalibration:
    def __init__(self, C=1e6, class_weight='balanced', max_iter=1000, solver='lbfgs'):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.solver = solver
        self.model = None
        self._fit_meta = {}

    def _transform(self, logits):
        logits = np.asarray(logits)
        logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        probs = 1 / (1 + np.exp(-logits))
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        X = np.column_stack([np.log(probs), np.log(1.0 - probs)])
        return X

    def fit(self, logits, labels):
        from sklearn.linear_model import LogisticRegression
        X = self._transform(logits)
        clf = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter, solver=self.solver)
        clf.fit(X, labels)
        self.model = clf
        labels_arr = np.asarray(labels)
        pos_ratio = float(np.mean(labels_arr)) if len(labels_arr) > 0 else 0.0
        self._fit_meta = {
            'n_samples': int(len(labels_arr)),
            'train_positive_ratio': pos_ratio,
            'fit_time': pd.Timestamp.now().isoformat()
        }

    def predict_proba(self, logits):
        if self.model is None:
            raise RuntimeError("BetaCalibration not fitted, call fit() first.")
        X = self._transform(logits)
        return self.model.predict_proba(X)[:, 1]

    def get_params(self):
        if self.model is not None and hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            a = float(self.model.coef_[0][0])
            b = float(self.model.coef_[0][1])
            c = float(self.model.intercept_[0])
            return {'a': a, 'b': b, 'c': c}
        return None

    def save(self, path, method: str = 'beta', model_type: str | None = None, fold: int | None = None):
        params = self.get_params() or {}
        data = {
            'method': method,
            'version': 1,
            'params': {k: float(v) for k, v in params.items()},
            'meta': {
                **self._fit_meta,
                'model_type': model_type,
                'fold': fold
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'params' in data:
            params = data['params']
            if 'a' in params and 'b' in params and 'c' in params:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(C=self.C, class_weight=self.class_weight, 
                                               max_iter=self.max_iter, solver=self.solver)
                dummy_X = np.array([[0.1, -0.1], [0.2, -0.2]])
                dummy_y = np.array([0, 1])
                self.model.fit(dummy_X, dummy_y)
                self.model.coef_ = np.array([[params['a'], params['b']]])
                self.model.intercept_ = np.array([params['c']])
                self._fit_meta = data.get('meta', {})
                print(f"Loaded Beta params: a={params['a']:.4f}, b={params['b']:.4f}, c={params['c']:.4f}")
        elif isinstance(data, dict) and hasattr(data, 'model'):
            self.model = data.model
            print("Loaded legacy Beta model")
        else:
            self.model = data
            print("Loaded legacy Beta model")
    
    @classmethod
    def from_dict(cls, data: dict):
        calibrator = cls()
        params = data['params']
        if 'a' in params and 'b' in params and 'c' in params:
            from sklearn.linear_model import LogisticRegression
            calibrator.model = LogisticRegression(C=calibrator.C, class_weight=calibrator.class_weight,
                                                 max_iter=calibrator.max_iter, solver=calibrator.solver)
            dummy_X = np.array([[0.1, -0.1], [0.2, -0.2]])
            dummy_y = np.array([0, 1])
            calibrator.model.fit(dummy_X, dummy_y)
            calibrator.model.coef_ = np.array([[params['a'], params['b']]])
            calibrator.model.intercept_ = np.array([params['c']])
            calibrator._fit_meta = data.get('meta', {})
        return calibrator

class TemperatureScaling:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self._fit_meta = {}

    def fit(self, logits, labels):
        from scipy.optimize import minimize_scalar
        
        def loss(temp):
            scaled_logits = logits / temp
            probs = 1 / (1 + np.exp(-scaled_logits))
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        
        result = minimize_scalar(loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        print(f"Optimized temperature: {self.temperature:.4f}")
        labels_arr = np.asarray(labels)
        pos_ratio = float(np.mean(labels_arr)) if len(labels_arr) > 0 else 0.0
        self._fit_meta = {
            'n_samples': int(len(labels_arr)),
            'train_positive_ratio': pos_ratio,
            'fit_time': pd.Timestamp.now().isoformat()
        }

    def predict_proba(self, logits):
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def get_params(self):
        return {'temperature': self.temperature}

    def save(self, path, method: str = 'temperature', model_type: str | None = None, fold: int | None = None):
        data = {
            'method': method,
            'version': 1,
            'params': {
                'temperature': float(self.temperature)
            },
            'meta': {
                **self._fit_meta,
                'model_type': model_type,
                'fold': fold
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'params' in data:
            self.temperature = data['params']['temperature']
            self._fit_meta = data.get('meta', {})
            print(f"Loaded Temperature params: T={self.temperature:.4f}")
        elif hasattr(data, 'temperature'):
            self.temperature = data.temperature
            print("Loaded legacy Temperature model")
        else:
            raise ValueError("Unrecognized Temperature calibrator format")
    
    @classmethod
    def from_dict(cls, data: dict):
        temperature = data['params']['temperature']
        calibrator = cls(temperature=temperature)
        calibrator._fit_meta = data.get('meta', {})
        return calibrator


class IsotonicCalibration:
    def __init__(self, out_of_bounds='clip'):
        from sklearn.isotonic import IsotonicRegression
        self.model = IsotonicRegression(out_of_bounds=out_of_bounds)
        self._fit_meta = {}
        self._iso_points = None  

    def fit(self, logits, labels):
        probs = 1 / (1 + np.exp(-logits))
        self.model.fit(probs, labels)
        labels_arr = np.asarray(labels)
        pos_ratio = float(np.mean(labels_arr)) if len(labels_arr) > 0 else 0.0
        self._fit_meta = {
            'n_samples': int(len(labels_arr)),
            'train_positive_ratio': pos_ratio,
            'fit_time': pd.Timestamp.now().isoformat()
        }
        x_points, y_points = None, None
        if hasattr(self.model, 'X_thresholds_') and hasattr(self.model, 'y_thresholds_'):
            x_points = getattr(self.model, 'X_thresholds_')
            y_points = getattr(self.model, 'y_thresholds_')
        elif hasattr(self.model, 'X_') and hasattr(self.model, 'y_'):
            x_points = getattr(self.model, 'X_')
            y_points = getattr(self.model, 'y_')
        else:
            grid = np.linspace(0.0, 1.0, num=256).astype(float)
            y_pred = self.model.predict(grid)
            x_points, y_points = grid, y_pred
        self._iso_points = (np.asarray(x_points, dtype=float), np.asarray(y_points, dtype=float))

    def predict_proba(self, logits):
        probs = 1 / (1 + np.exp(-logits))
        return self.model.predict(probs)

    def get_params(self):
        if self._iso_points:
            return {'x': self._iso_points[0].tolist(), 'y': self._iso_points[1].tolist()}
        return None

    def save(self, path, method: str = 'isotonic', model_type: str | None = None, fold: int | None = None):
        if self._iso_points is None:
            grid = np.linspace(0.0, 1.0, num=256).astype(float)
            y_pred = self.model.predict(grid)
            x_points, y_points = grid, y_pred
        else:
            x_points, y_points = self._iso_points
        data = {
            'method': method,
            'version': 1,
            'params': {
                'x': np.asarray(x_points, dtype=float).tolist(),
                'y': np.asarray(y_points, dtype=float).tolist(),
                'out_of_bounds': 'clip'  
            },
            'meta': {
                **self._fit_meta,
                'model_type': model_type,
                'fold': fold
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'params' in data:
            params = data['params']
            if 'x' in params and 'y' in params:
                from sklearn.isotonic import IsotonicRegression
                x_points = np.array(params['x'])
                y_points = np.array(params['y'])
                
                out_of_bounds = params.get('out_of_bounds', 'clip')
                self.model = IsotonicRegression(out_of_bounds=out_of_bounds)
                self.model.fit(x_points, y_points)
                self._iso_points = (x_points, y_points)
                self._fit_meta = data.get('meta', {})
                print(f"Loaded Isotonic mapping points: {len(x_points)} points")
        elif hasattr(data, 'model'):
            self.model = data.model
            print("Loaded legacy Isotonic model")
        else:
            self.model = data
            print("Loaded legacy Isotonic model")
    
    @classmethod
    def from_dict(cls, data: dict):
        params = data['params']
        out_of_bounds = params.get('out_of_bounds', 'clip')
        calibrator = cls(out_of_bounds=out_of_bounds)
        
        if 'x' in params and 'y' in params:
            from sklearn.isotonic import IsotonicRegression
            x_points = np.array(params['x'])
            y_points = np.array(params['y'])
            
            calibrator.model = IsotonicRegression(out_of_bounds=out_of_bounds)
            calibrator.model.fit(x_points, y_points)
            calibrator._iso_points = (x_points, y_points)
            calibrator._fit_meta = data.get('meta', {})
        
        return calibrator

class BoostingCalibration:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_leaf=50, subsample=0.8, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.model = None
        self._fit_meta = {}

    def fit(self, logits, labels):
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=0
        )
        
        self.model.fit(logits.reshape(-1, 1), labels)
        
        labels_arr = np.asarray(labels)
        pos_ratio = float(np.mean(labels_arr)) if len(labels_arr) > 0 else 0.0
        self._fit_meta = {
            'n_samples': int(len(labels_arr)),
            'train_positive_ratio': pos_ratio,
            'fit_time': pd.Timestamp.now().isoformat()
        }

    def predict_proba(self, logits):
        if self.model is None:
            raise RuntimeError("BoostingCalibration not fitted, call fit() first.")
        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]

    def get_params(self):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample,
            'random_state': self.random_state
        }

    def save(self, path, method: str = 'boosting', model_type: str | None = None, fold: int | None = None):
        params = self.get_params()

        data = {
            'method': method,
            'version': 1,
            'params': params,
            'model': self.model, 
            'meta': {
                **self._fit_meta,
                'model_type': model_type,
                'fold': fold
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'params' in data:
            params = data.get('params', {})
            self.n_estimators = params.get('n_estimators', self.n_estimators)
            self.learning_rate = params.get('learning_rate', self.learning_rate)
            self.max_depth = params.get('max_depth', self.max_depth)
            self.min_samples_leaf = params.get('min_samples_leaf', self.min_samples_leaf)
            self.subsample = params.get('subsample', self.subsample)
            self.random_state = params.get('random_state', self.random_state)
            self._fit_meta = data.get('meta', {})

            if 'model' in data:
                self.model = data['model']
                print(f"Loaded Boosting calibrator (full model): {params}")
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    subsample=self.subsample,
                    random_state=self.random_state,
                    verbose=0
                )
                print(f"Loaded Boosting calibrator params (requires retraining): {params}")

        elif isinstance(data, dict) and 'model' in data:
            self.model = data['model']
            self._fit_meta = data.get('meta', {})
            params = data.get('params', {})
            self.n_estimators = params.get('n_estimators', self.n_estimators)
            self.learning_rate = params.get('learning_rate', self.learning_rate)
            self.max_depth = params.get('max_depth', self.max_depth)
            self.min_samples_leaf = params.get('min_samples_leaf', self.min_samples_leaf)
            self.subsample = params.get('subsample', self.subsample)
            self.random_state = params.get('random_state', self.random_state)
            print(f"Loaded legacy Boosting calibrator: n_estimators={self.n_estimators}, learning_rate={self.learning_rate}")
        else:
            raise ValueError("Unrecognized Boosting calibrator format")
    
    @classmethod
    def from_dict(cls, data: dict):
        params = data.get('params', {})
        calibrator = cls(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            min_samples_leaf=params.get('min_samples_leaf', 50),
            subsample=params.get('subsample', 0.8),
            random_state=params.get('random_state', 42)
        )
        calibrator.model = data.get('model')
        calibrator._fit_meta = data.get('meta', {})
        return calibrator

def tune_beta_calibration_cv(logits, labels, C_values=None, cv_folds=3):
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    
    if C_values is None:
        C_values = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    best_score = float('inf')
    best_params = {'C': 1e6}
    
    print(f"Starting Beta hyperparameter tuning (inner {cv_folds}-fold CV on training set to avoid leakage)...")
    
    for C in C_values:
        cv_scores = []
        for train_idx, val_idx in skf.split(logits, labels):
            try:
                train_logits_cv, val_logits_cv = logits[train_idx], logits[val_idx]
                train_labels_cv, val_labels_cv = labels[train_idx], labels[val_idx]
                
                calibrator = BetaCalibration(C=C)
                calibrator.fit(train_logits_cv, train_labels_cv)
                val_probs = calibrator.predict_proba(val_logits_cv)
                score = log_loss(val_labels_cv, val_probs)
                cv_scores.append(score)
                
            except Exception as e:
                print(f"  CV tuning failed C={C}: {e}")
                continue
        
        if cv_scores:
            avg_score = np.mean(cv_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = {'C': C}
    
    print(f"✅ Best Beta params (inner CV): C={best_params['C']}, avg_log_loss={best_score:.4f}")
    
    return best_params

def tune_temperature_scaling_cv(logits, labels, temp_range=None, cv_folds=3):
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    
    if temp_range is None:
        temp_range = (0.1, 10.0)  # temperature range
    
    print(f"Starting Temperature hyperparameter tuning (inner {cv_folds}-fold CV on training set to avoid leakage)...")
    print("Note: Temperature scaling is solved by optimization; no extra hyperparameter tuning needed.")
    
    return {}

def tune_isotonic_calibration_cv(logits, labels, out_of_bounds_values=None, cv_folds=3):
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    
    if out_of_bounds_values is None:
        out_of_bounds_values = ['clip']
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    best_score = float('inf')
    best_params = {'out_of_bounds': 'clip'}
    
    print(f"Starting Isotonic hyperparameter tuning (inner {cv_folds}-fold CV on training set to avoid leakage)...")
    
    for out_of_bounds in out_of_bounds_values:
        cv_scores = []
        for train_idx, val_idx in skf.split(logits, labels):
            try:
                train_logits_cv, val_logits_cv = logits[train_idx], logits[val_idx]
                train_labels_cv, val_labels_cv = labels[train_idx], labels[val_idx]
                
                calibrator = IsotonicCalibration(out_of_bounds=out_of_bounds)
                calibrator.fit(train_logits_cv, train_labels_cv)
                val_probs = calibrator.predict_proba(val_logits_cv)
                score = log_loss(val_labels_cv, val_probs)
                cv_scores.append(score)
                
            except Exception as e:
                print(f"  CV tuning failed out_of_bounds={out_of_bounds}: {e}")
                continue
        
        if cv_scores:
            avg_score = np.mean(cv_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = {'out_of_bounds': out_of_bounds}
    
    print(f"✅ Best Isotonic params (inner CV): out_of_bounds={best_params['out_of_bounds']}, avg_log_loss={best_score:.4f}")
    
    return best_params

def tune_platt_scaling_cv(logits, labels, C_values=None, class_weight_values=None, cv_folds=3):
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    if class_weight_values is None:
        class_weight_values = ['balanced', None] 
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    best_score = float('inf')
    best_params = {'C': 1.0, 'class_weight': 'balanced'}
    
    print(f"Starting Platt hyperparameter tuning (inner {cv_folds}-fold CV on training set to avoid leakage)...")
    
    for C in C_values:
        for class_weight in class_weight_values:
            cv_scores = []
            for train_idx, val_idx in skf.split(logits, labels):
                try:
                    train_logits_cv, val_logits_cv = logits[train_idx], logits[val_idx]
                    train_labels_cv, val_labels_cv = labels[train_idx], labels[val_idx]
                    
                    calibrator = PlattScaling(C=C, class_weight=class_weight)
                    calibrator.fit(train_logits_cv, train_labels_cv)
                    val_probs = calibrator.predict_proba(val_logits_cv)
                    score = log_loss(val_labels_cv, val_probs)
                    cv_scores.append(score)
                    
                except Exception as e:
                    print(f"  CV tuning failed C={C}, class_weight={class_weight}: {e}")
                    continue
            
            if cv_scores:
                avg_score = np.mean(cv_scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'C': C, 'class_weight': class_weight}
    
    print(f"✅ Best Platt params (inner CV): C={best_params['C']}, class_weight={best_params['class_weight']}, avg_log_loss={best_score:.4f}")
    
    return best_params

def tune_boosting_calibration_cv(logits, labels, cv_folds=3):
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    
    test_configs = [
        {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 2, 'min_samples_leaf': 50},
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 50},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 30},
        {'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 3, 'min_samples_leaf': 100},
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 2, 'min_samples_leaf': 100},
    ]
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    best_score = float('inf')
    best_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_leaf': 50
    }
    
    print(f"Starting Boosting hyperparameter tuning (inner {cv_folds}-fold CV on training set to avoid leakage)...")
    print(f"Testing {len(test_configs)} parameter configs...")
    
    for params in test_configs:
        cv_scores = []
        for train_idx, val_idx in skf.split(logits, labels):
            try:
                train_logits_cv, val_logits_cv = logits[train_idx], logits[val_idx]
                train_labels_cv, val_labels_cv = labels[train_idx], labels[val_idx]
                
                calibrator = BoostingCalibration(**params)
                calibrator.fit(train_logits_cv, train_labels_cv)
                val_probs = calibrator.predict_proba(val_logits_cv)
                score = log_loss(val_labels_cv, val_probs)
                cv_scores.append(score)
                
            except Exception as e:
                print(f"  CV tuning failed {params}: {e}")
                continue
        
        if cv_scores:
            avg_score = np.mean(cv_scores)
            print(f"  Params: {params}, avg_log_loss={avg_score:.4f}")
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
    
    print(f"✅ Best Boosting params (inner CV): {best_params}, avg_log_loss={best_score:.4f}")
    
    return best_params

def save_logits_to_txt(protein_pairs, labels, logits, fold, model_type, output_dir):
    txt_file = os.path.join(output_dir, f"logits_data_fold{fold}_{model_type}.txt")
    
    try:
        with open(txt_file, 'w') as f:
            f.write("# Protein1\tProtein2\tLabel\tlogits\n")
            for (p1, p2), label, logit in zip(protein_pairs, labels, logits):
                f.write(f"{p1}\t{p2}\t{label}\t{logit:.6f}\n")
        
        print(f"Saved logits data to: {txt_file} (samples: {len(protein_pairs)})")
    except Exception as e:
        print(f"Failed to save logits data: {e}")

def save_calibrated_probs_to_txt(protein_pairs, labels, logits, probs, fold, model_type, method, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    txt_file = os.path.join(output_dir, f"calibrated_probs_{method}_fold{fold}_{model_type}_{dataset_name}.txt")
    try:
        with open(txt_file, 'w') as f:
            f.write("# Protein1\tProtein2\tLabel\tlogit\tcalibrated_prob\n")
            for (p1, p2), y, logit, p in zip(protein_pairs, labels, logits, probs):
                f.write(f"{p1}\t{p2}\t{int(y)}\t{float(logit):.6f}\t{float(p):.6f}\n")
        print(f"Saved calibrated probabilities to: {txt_file} (samples: {len(labels)})")
    except Exception as e:
        print(f"Failed to save calibrated probabilities: {e}")

def clear_logits_cache(cache_dir):
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('_logits_cache.pkl')]
    if not cache_files:
        print(f"Cache directory is empty: {cache_dir}")
        return
    
    for cache_file in cache_files:
        file_path = os.path.join(cache_dir, cache_file)
        try:
            os.remove(file_path)
            print(f"Deleted cache file: {cache_file}")
        except Exception as e:
            print(f"Failed to delete cache file {cache_file}: {e}")
    
    print(f"Cleared {len(cache_files)} cache files")

def create_5fold_splits(data, n_splits=5, stratify=True, seed=42):
    if not stratify:
        np.random.seed(seed)
        np.random.shuffle(data)
        fold_size = len(data) // n_splits
        folds = []
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(data)
            folds.append(data[start_idx:end_idx])
        return folds

    pos_data = [item for item in data if item[2] == 1]
    neg_data = [item for item in data if item[2] == 0]

    print(f"Positive samples: {len(pos_data)}, Negative samples: {len(neg_data)}")

    rng = np.random.default_rng(seed)
    rng.shuffle(pos_data)
    rng.shuffle(neg_data)

    pos_per_fold = len(pos_data) // n_splits
    neg_per_fold = len(neg_data) // n_splits

    folds = []
    pos_start, neg_start = 0, 0

    for i in range(n_splits):
        if i == n_splits - 1:
            pos_end = len(pos_data)
            neg_end = len(neg_data)
        else:
            pos_end = pos_start + pos_per_fold
            neg_end = neg_start + neg_per_fold

        fold_pos = pos_data[pos_start:pos_end]
        fold_neg = neg_data[neg_start:neg_end]

        fold_data = fold_pos + fold_neg
        rng.shuffle(fold_data)
        folds.append(fold_data)

        pos_start, neg_start = pos_end, neg_end

        print(f"Fold {i+1}: {len(fold_data)} samples (pos {len(fold_pos)}, neg {len(fold_neg)})")

    return folds

def save_fold_data_to_txt(fold_data, fold_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, f"fold_{fold_idx}_train.txt")
    with open(train_file, 'w') as f:
        f.write("# Protein1\tProtein2\tLabel\n")
        for p1, p2, label in fold_data['train']:
            f.write(f"{p1}\t{p2}\t{label}\n")
    val_file = os.path.join(output_dir, f"fold_{fold_idx}_val.txt")
    with open(val_file, 'w') as f:
        f.write("# Protein1\tProtein2\tLabel\n")
        for p1, p2, label in fold_data['val']:
            f.write(f"{p1}\t{p2}\t{label}\n")

    print(f"Saved Fold {fold_idx} data: train {len(fold_data['train'])}, val {len(fold_data['val'])}")

def get_fold_train_val_data(folds, val_fold_idx):
    val_data = folds[val_fold_idx]
    train_data = []
    for i, fold in enumerate(folds):
        if i != val_fold_idx:
            train_data.extend(fold)

    return train_data, val_data

def load_wet_experiment_data(pos_file: str, neg_file: str, split_ratios: Optional[Tuple[float, float, float]] = None, stratify: bool = True, seed: int = 42):
    pos_data = []
    neg_data = []
    
    if os.path.exists(pos_file):
        with open(pos_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    p1, p2 = parts[0], parts[1]
                    pos_data.append((p1, p2, 1)) 

    if os.path.exists(neg_file):
        with open(neg_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    p1, p2 = parts[0], parts[1]
                    neg_data.append((p1, p2, 0))  
    
    all_data = pos_data + neg_data
    print(f"Loaded wet-lab data: pos {len(pos_data)}, neg {len(neg_data)}, total {len(all_data)}")
    
    if split_ratios is None:
        return all_data

    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1"

    if not stratify:
        np.random.seed(seed)
        np.random.shuffle(all_data)
        n_total = len(all_data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        test_data = all_data[n_train + n_val:]
    else:
        rng = np.random.default_rng(seed)
        pos_idx = np.arange(len(pos_data))
        neg_idx = np.arange(len(neg_data))
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        n_pos_total = len(pos_data)
        n_neg_total = len(neg_data)

        n_pos_train = int(n_pos_total * train_ratio)
        n_pos_val = int(n_pos_total * val_ratio)
        n_pos_test = n_pos_total - n_pos_train - n_pos_val

        n_neg_train = int(n_neg_total * train_ratio)
        n_neg_val = int(n_neg_total * val_ratio)
        n_neg_test = n_neg_total - n_neg_train - n_neg_val

        pos_train = [pos_data[i] for i in pos_idx[:n_pos_train]]
        pos_val = [pos_data[i] for i in pos_idx[n_pos_train:n_pos_train + n_pos_val]]
        pos_test = [pos_data[i] for i in pos_idx[n_pos_train + n_pos_val:]]

        neg_train = [neg_data[i] for i in neg_idx[:n_neg_train]]
        neg_val = [neg_data[i] for i in neg_idx[n_neg_train:n_neg_train + n_neg_val]]
        neg_test = [neg_data[i] for i in neg_idx[n_neg_train + n_neg_val:]]

        train_data = pos_train + neg_train
        val_data = pos_val + neg_val
        test_data = pos_test + neg_test

        rng.shuffle(train_data)
        rng.shuffle(val_data)
        rng.shuffle(test_data)

          print(f"Data split (stratified): train {len(train_data)} (pos {len(pos_train)}, neg {len(neg_train)}), "
              f"val {len(val_data)} (pos {len(pos_val)}, neg {len(neg_val)}), "
              f"test {len(test_data)} (pos {len(pos_test)}, neg {len(neg_test)})")

    return train_data, val_data, test_data

def load_logits_cache(data_hash, model_type, cache_dir, model_key: str | None = None):
    suffix = f"_{model_key}" if model_key else ""
    cache_file = os.path.join(cache_dir, f"wet_experiment_logits_cache_{data_hash}_{model_type}{suffix}.pkl")

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        print(f"Loaded logits cache: {cache_file} (samples: {cache_data['num_samples']})")
        return cache_data['logits'], cache_data['labels'], cache_data.get('protein_pairs', [])
    except Exception as e:
        print(f"Failed to load cache {cache_file}: {e}")
        return None

def load_fold_logits_cache(fold, model_type, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"fold_{fold}_{model_type}_logits_cache.pkl")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Loaded fold cache: {cache_file} (samples: {cache_data['num_samples']})")
        return cache_data['logits'], cache_data['labels'], cache_data.get('protein_pairs', [])
    except Exception as e:
        print(f"Failed to load fold cache {cache_file}: {e}")
        return None

def save_logits_cache(data_hash, model_type, logits, labels, protein_pairs, cache_dir, model_key: str | None = None):
    os.makedirs(cache_dir, exist_ok=True)
    suffix = f"_{model_key}" if model_key else ""
    cache_file = os.path.join(cache_dir, f"wet_experiment_logits_cache_{data_hash}_{model_type}{suffix}.pkl")
    cache_data = {
        'logits': logits,
        'labels': labels,
        'protein_pairs': protein_pairs,
        'num_samples': len(logits),
        'data_hash': data_hash,
        'model_type': model_type,
        'timestamp': str(pd.Timestamp.now())
    }
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved logits cache: {cache_file} (samples: {len(cache_data['logits'])})")
    except Exception as e:
        print(f"Failed to save cache {cache_file}: {e}")


def save_fold_logits_cache(fold, model_type, logits, labels, protein_pairs, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"fold_{fold}_{model_type}_logits_cache.pkl")
    cache_data = {
        'logits': logits,
        'labels': labels,
        'protein_pairs': protein_pairs,
        'num_samples': len(logits),
        'fold': fold,
        'model_type': model_type,
        'timestamp': str(pd.Timestamp.now())
    }
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved fold logits cache: {cache_file} (samples: {len(cache_data['logits'])})")
    except Exception as e:
        print(f"Failed to save fold cache {cache_file}: {e}")

def get_data_hash(data):
    import hashlib
    data_str = str(sorted(data)) 
    return hashlib.md5(data_str.encode()).hexdigest()[:16]

def get_logits_from_model_lazy(model, val_data, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model.to(device)

    valid_samples = []
    for p1, p2, label in val_data:
        if check_protein_exists(p1) and check_protein_exists(p2):
            valid_samples.append((p1, p2, label))

    print(f"Valid samples: {len(valid_samples)}/{len(val_data)}")

    for i in tqdm(range(0, len(valid_samples), batch_size), desc="Batching logits"):
        batch_samples = valid_samples[i:i + batch_size]
        batch_logits = []
        batch_labels = []

        batch_graphs = []
        batch_valid_indices = []

        for j, (p1, p2, label) in enumerate(batch_samples):
            try:
                g1, g2 = getData_GCN(p1, p2)
                g1 = g1.to(device)
                g2 = g2.to(device)
                batch_graphs.append((g1, g2))
                batch_valid_indices.append(j)
                batch_labels.append(label)
            except Exception as e:
                print(f"Skipped {p1}-{p2}: {e}")
                continue

        if not batch_graphs:
            continue

        try:
            with torch.no_grad():
                batch_results = []
                for g1, g2 in batch_graphs:
                    prob_logits = model(g1, g2, g1.ndata['fea'], g2.ndata['fea'])
                    logit = prob_logits[:, 1].cpu().item()
                    batch_results.append(logit)

                batch_logits = np.array(batch_results)

        except Exception as e:
            print(f"Batch processing failed: {e}")
            batch_logits = []
            batch_labels = []
            for (g1, g2), label in zip(batch_graphs, batch_labels):
                try:
                    with torch.no_grad():
                        prob_logits = model(g1, g2, g1.ndata['fea'], g2.ndata['fea'])
                        logit = prob_logits[:, 1].cpu().item()
                        batch_logits.append(logit)
                        batch_labels.append(label)
                except Exception as e2:
                    print(f"Single-sample processing failed: {e2}")
                    continue
            batch_logits = np.array(batch_logits)

        if len(batch_logits) > 0:
            yield batch_logits, np.array(batch_labels)

def get_logits_from_model_batch(model, val_data, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu', cache_dir=None, model_type='unknown', model_key=None):
    if cache_dir is not None and model_type != 'unknown':
        data_hash = get_data_hash(val_data)
        cached_result = load_logits_cache(data_hash, model_type, cache_dir, model_key)
        if cached_result is not None:
            print(f"Loaded logits from cache: {len(cached_result[0])} samples")
            return cached_result

    try:
        import dgl
    except ImportError:
        print("DGL not installed, falling back to standard batching")
        return get_logits_from_model_lazy(model, val_data, batch_size, device)

    model.eval()
    model.to(device)

    valid_samples = []
    for p1, p2, label in val_data:
        if check_protein_exists(p1) and check_protein_exists(p2):
            valid_samples.append((p1, p2, label))

    print(f"Valid samples: {len(valid_samples)}/{len(val_data)}")

    all_logits = []
    all_labels = []
    all_protein_pairs = []  
    for i in tqdm(range(0, len(valid_samples), batch_size), desc="Batching logits"):
        batch_samples = valid_samples[i:i + batch_size]

        batch_g1 = []
        batch_g2 = []
        batch_labels = []
        batch_protein_pairs = []

        for p1, p2, label in batch_samples:
            try:
                g1, g2 = getData_GCN(p1, p2)
                batch_g1.append(g1)
                batch_g2.append(g2)
                batch_labels.append(label)
                batch_protein_pairs.append((p1, p2))  
            except Exception as e:
                print(f"Skipped {p1}-{p2}: {e}")
                continue

        if not batch_g1:
            continue

        try:
            batched_g1 = dgl.batch(batch_g1).to(device)
            batched_g2 = dgl.batch(batch_g2).to(device)

            with torch.no_grad():
                prob_logits = model(batched_g1, batched_g2, batched_g1.ndata['fea'], batched_g2.ndata['fea'])

                if prob_logits.dim() == 2 and prob_logits.shape[0] == len(batch_g1)
                    batch_logits = prob_logits[:, 1].cpu().numpy()
                else:
                    batch_logits = []
                    for j in range(len(batch_g1)):
                        logit = prob_logits[j, 1].cpu().item()
                        batch_logits.append(logit)
                    batch_logits = np.array(batch_logits)

            all_logits.extend(batch_logits)
            all_labels.extend(batch_labels)
            all_protein_pairs.extend(batch_protein_pairs)

        except Exception as e:
            print(f"DGL batch failed, falling back to per-sample processing: {e}")
            for (p1, p2), g1, g2, label in zip(batch_protein_pairs, batch_g1, batch_g2, batch_labels):
                try:
                    g1 = g1.to(device)
                    g2 = g2.to(device)
                    with torch.no_grad():
                        prob_logits = model(g1, g2, g1.ndata['fea'], g2.ndata['fea'])
                        logit = prob_logits[:, 1].cpu().item()
                        all_logits.append(logit)
                        all_labels.append(label)
                        all_protein_pairs.append((p1, p2))
                except Exception as e2:
                    print(f"Single-sample processing failed: {e2}")
                    continue

    return np.array(all_logits), np.array(all_labels), all_protein_pairs

def optimize_threshold(y_true, y_prob, min_precision=0.82):
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_indices = precisions[:-1] >= min_precision
    
    if np.any(valid_indices):
        valid_recalls = recalls[:-1][valid_indices]
        valid_thresholds = thresholds[valid_indices]
        valid_precisions = precisions[:-1][valid_indices]
        best_idx = np.argmax(valid_recalls)
        
        optimal_threshold = valid_thresholds[best_idx]
        precision_at_threshold = valid_precisions[best_idx]
        recall_at_threshold = valid_recalls[best_idx]
        
        print(f"Found optimal threshold: {optimal_threshold:.4f}, Precision: {precision_at_threshold:.4f}, Recall: {recall_at_threshold:.4f}")
        
        return optimal_threshold, precision_at_threshold, recall_at_threshold, False
    else:
        print(f"No threshold meets precision >= {min_precision}; using default 0.5")
        return 0.5, None, None, True

def evaluate_calibration(y_true, y_prob, method_name, fold, output_dir, optimal_threshold=None, dataset_type='val'):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
    
    threshold_fallback = False 
    
    if optimal_threshold is None:
        if dataset_type == 'val':
            optimal_threshold, opt_precision, opt_recall, threshold_fallback = optimize_threshold(y_true, y_prob, min_precision=0.82)
            threshold_optimized = True
            if threshold_fallback:
                print(f"  ⚠️ Validation set did not find a matching threshold; falling back to 0.5")
            else:
                print(f"  ✅ Validation set optimized threshold: {optimal_threshold:.4f} (Precision: {opt_precision:.4f}, Recall: {opt_recall:.4f})")
        else:
            optimal_threshold = 0.5
            opt_precision = None
            opt_recall = None
            threshold_optimized = False
            threshold_fallback = True
            print(f"  ⚠️ Warning: {dataset_type} dataset should not optimize threshold; using 0.5")
    else:
        y_pred_temp = (y_prob >= optimal_threshold).astype(int)
        opt_precision = precision_score(y_true, y_pred_temp, zero_division=0)
        opt_recall = recall_score(y_true, y_pred_temp, zero_division=0)
        threshold_optimized = False
        if dataset_type == 'test':
            print(f"  ✅ Test set uses validation threshold: {optimal_threshold:.4f} (Precision: {opt_precision:.4f}, Recall: {opt_recall:.4f})")

    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{method_name} - Fold {fold}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title(f'Reliability Diagram - {method_name} Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'calibration_plot_{method_name}_fold{fold}.png'))
    plt.close()

    ece = 0
    for i in range(10):
        bin_mask = (y_prob >= i/10) & (y_prob < (i+1)/10)
        if np.sum(bin_mask) > 0:
            bin_prob = np.mean(y_prob[bin_mask])
            bin_true = np.mean(y_true[bin_mask])
            ece += np.abs(bin_prob - bin_true) * np.sum(bin_mask)
    ece /= len(y_true)

    brier = brier_score_loss(y_true, y_prob)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0 

    results = {
        'method': method_name,
        'fold': fold,
        'ece': float(ece),
        'brier': float(brier),
        'auroc': float(auroc),
        'auprc': float(auprc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc),
        'specificity': float(specificity),
        'npv': float(npv),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'optimal_threshold': float(optimal_threshold),
        'opt_precision': opt_precision,
        'opt_recall': opt_recall,
        'threshold_fallback': threshold_fallback,
    }

    txt_file = os.path.join(output_dir, f'calibration_results_{method_name}_fold{fold}.txt')
    with open(txt_file, 'w') as f:
        f.write(f"Calibration method: {method_name}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Sample count: {len(y_true)}\n")
        f.write("-" * 50 + "\n")
        if threshold_optimized:
            f.write("Threshold tuning:\n")
            f.write(f"  Tuned threshold: {optimal_threshold:.4f}\n")
            if opt_precision is not None:
                f.write(f"  Precision at tuned threshold: {opt_precision:.4f}\n")
                f.write(f"  Recall at tuned threshold: {opt_recall:.4f}\n")
            else:
                f.write("  Using default threshold 0.5\n")
        else:
            f.write("Using provided threshold:\n")
            f.write(f"  Threshold: {optimal_threshold:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write("Calibration metrics:\n")
        f.write(f"  ECE (Expected Calibration Error): {ece:.4f}\n")
        f.write(f"  Brier Score: {brier:.4f}\n")
        f.write(f"  AUROC: {auroc:.4f}\n")
        f.write(f"  AUPRC: {auprc:.4f}\n")
        f.write("\nClassification metrics (using tuned threshold):\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
        f.write(f"  MCC (Matthews Correlation Coefficient): {mcc:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n")
        f.write(f"  NPV (Negative Predictive Value): {npv:.4f}\n")
        f.write("\nConfusion matrix:\n")
        f.write(f"  True Positive (TP): {tp}\n")
        f.write(f"  False Positive (FP): {fp}\n")
        f.write(f"  True Negative (TN): {tn}\n")
        f.write(f"  False Negative (FN): {fn}\n")
        f.write(f"\nTotal correct predictions: {tp + tn}/{len(y_true)} ({(tp + tn)/len(y_true)*100:.1f}%)\n")

    return results

def calibrate_and_save_models_kfold(model_types=['auprc', 'auroc'], folds=None, batch_size=64, use_lazy_loading=True, cache_dir=None,
                          use_wet_experiment_data=False, wet_pos_file=None, wet_neg_file=None, do_kfold=True, do_final=False):
    if model_types is None:
        model_types = ['auprc', 'auroc', 'f1']
    elif isinstance(model_types, str):
        model_types = [model_types]

    if folds is None:
        folds = list(range(1, 6))
    elif isinstance(folds, int):
        folds = [folds]

    base_dir = "LNGCN_main/results/balance_human"

    base_output_dir = "LNGCN_main/results/calibrate"
    shuju_fold_dir = os.path.join(base_output_dir, "data_fold") 
    fold_jiaozhun_dir = os.path.join(base_output_dir, "evaluate_summary") 
    fold_pingjia_dir = os.path.join(fold_jiaozhun_dir, "pingjia") 
    ceshi_jiaozhun_dir = os.path.join(base_output_dir, "test")
    final_jiaozhun_dir = os.path.join(base_output_dir, "val")
    lanjz_dir = os.path.join(base_output_dir, "lanjz") 

    methods = ['platt', 'beta', 'temperature', 'isotonic', 'boosting']

    print(f"Will calibrate these model types: {', '.join(model_types).upper()}")
    print(f"Will process these folds: {folds}")
    print(f"Use wet-lab data: {use_wet_experiment_data}")
    print(f"Run 5-fold cross-validation: {do_kfold}")
    print(f"Run final calibrator: {do_final}")

    if use_wet_experiment_data:
        if wet_pos_file is None or wet_neg_file is None:
            wet_pos_file = "LNGCN_main/data/calibration/3000_pos.txt"
            wet_neg_file = "LNGCN_main/data/calibration/3000_neg.txt"
        print(f"Positive file: {wet_pos_file}")
        print(f"Negative file: {wet_neg_file}")

    if cache_dir is None:
        cache_dir = os.path.join(base_output_dir, "logits_cache")

    if use_wet_experiment_data:
        train_val_data, _, test_data = load_wet_experiment_data(
            wet_pos_file, wet_neg_file, 
            split_ratios=(0.8, 0.0, 0.2), 
            stratify=True,
            seed=42
        )
        print(f"Data split complete: train+val {len(train_val_data)} samples, test {len(test_data)} samples")

        if do_kfold:
            folds_data = create_5fold_splits(train_val_data, n_splits=5, stratify=True, seed=42)
            print("Created 5-fold CV splits (based on train+val data)")

    kfold_results = {model_type: [] for model_type in model_types}
    final_results = {model_type: [] for model_type in model_types}

    if do_kfold and use_wet_experiment_data:
        print("\n" + "="*80)
        print("Starting 5-fold CV calibration")
        print("="*80)

        for fold_idx in range(5): 
            fold_num = fold_idx + 1
            print(f"\nProcessing Fold {fold_num} (validation set)")
            train_data, val_data = get_fold_train_val_data(folds_data, fold_idx)
            fold_data_dict = {'train': train_data, 'val': val_data}
            save_fold_data_to_txt(fold_data_dict, fold_num, shuju_fold_dir)
            for model_type in model_types:
                print(f"\n--- Processing {model_type.upper()} model (Fold {fold_num}) ---")
                output_dir = os.path.join(fold_jiaozhun_dir, f"calibrated_models_{model_type}")
                os.makedirs(output_dir, exist_ok=True)

                model_path = os.path.join(base_dir, f"fold{fold_num}_best_{model_type}.pt")
                if not os.path.exists(model_path):
                    print(f"Model file not found: {model_path}")
                    continue

                checkpoint = torch.load(model_path, map_location='cpu')
                actual_in_dim = checkpoint['cfc_preprocess.cells.0.backbone.0.weight'].shape[1]
                print(f"Checkpoint input dim: {actual_in_dim}")

                model = None
                for try_in_dim in [actual_in_dim + 1, actual_in_dim, actual_in_dim - 1, actual_in_dim - 2]:
                    try:
                        model = MyGCN(in_dim=try_in_dim, nhid=256, dropout=0.4, time_steps=5)
                        model.load_state_dict(checkpoint, strict=False)
                        print(f"Loaded model successfully with in_dim={try_in_dim}")
                        break
                    except RuntimeError as e:
                        print(f"Failed with in_dim={try_in_dim}: {str(e)[:100]}...")
                        continue
                else:
                    print(f"All in_dim attempts failed, skipping {model_type.upper()}")
                    continue

                model.eval()

                print(f"Computing training logits with {len(train_data)} samples")
                train_logits, train_labels, train_pairs = get_logits_from_model_batch(model, train_data, batch_size=batch_size, cache_dir=lanjz_dir, model_type=model_type, model_key=f"fold{fold_num}_train")

                if len(train_logits) == 0:
                    print(f"Fold {fold_num} has no valid training logits, skipping")
                    continue

                if not os.path.exists(os.path.join(lanjz_dir, f"wet_experiment_logits_cache_{get_data_hash(train_data)}_{model_type}_fold{fold_num}_train.pkl")):
                    save_logits_cache(get_data_hash(train_data), model_type, train_logits, train_labels, train_pairs, lanjz_dir, model_key=f"fold{fold_num}_train")
                    print(f"Saved training logits cache: fold{fold_num}_train")

                save_logits_to_txt(train_pairs, train_labels, train_logits, fold_num, model_type, output_dir)

                print(f"Computing validation logits for tuning with {len(val_data)} samples")
                val_logits, val_labels, val_pairs = get_logits_from_model_batch(model, val_data, batch_size=batch_size, cache_dir=lanjz_dir, model_type=model_type, model_key=f"fold{fold_num}_val")

                if len(val_logits) == 0:
                    print("No valid validation logits, skipping")
                    continue

                if not os.path.exists(os.path.join(lanjz_dir, f"wet_experiment_logits_cache_{get_data_hash(val_data)}_{model_type}_fold{fold_num}_val.pkl")):
                    save_logits_cache(get_data_hash(val_data), model_type, val_logits, val_labels, val_pairs, lanjz_dir, model_key=f"fold{fold_num}_val")
                    print(f"Saved validation logits cache: fold{fold_num}_val")

                for method in methods:
                    print(f"\n{'='*60}")
                    print(f"Calibration method: {method.upper()}")
                    print(f"{'='*60}")

                    if method == 'platt':
                        # Tune Platt params with inner CV on the training set
                        best_params = tune_platt_scaling_cv(train_logits, train_labels, cv_folds=3)
                        calibrator = PlattScaling(**best_params)
                    elif method == 'beta':
                        # Tune Beta params with inner CV on the training set
                        best_params = tune_beta_calibration_cv(train_logits, train_labels, cv_folds=3)
                        calibrator = BetaCalibration(**best_params)
                    elif method == 'temperature':
                        # Temperature scaling is determined by optimization; no CV tuning needed
                        tune_temperature_scaling_cv(train_logits, train_labels, cv_folds=3)
                        calibrator = TemperatureScaling()
                    elif method == 'isotonic':
                        # Tune Isotonic params with inner CV on the training set
                        best_params = tune_isotonic_calibration_cv(train_logits, train_labels, cv_folds=3)
                        calibrator = IsotonicCalibration(**best_params)
                    elif method == 'boosting':
                        # Tune Boosting params with inner CV on the training set
                        best_params = tune_boosting_calibration_cv(train_logits, train_labels, cv_folds=3)
                        calibrator = BoostingCalibration(**best_params)
                    else:
                        raise ValueError(f"Unknown calibration method: {method}")

                    print(f"Method implementation: {method} -> {type(calibrator).__name__}")

                    calibrator.fit(train_logits, train_labels)

                    cal_probs = calibrator.predict_proba(val_logits)

                    assert len(cal_probs) == len(val_labels), f"{method} output length mismatch: {len(cal_probs)} vs {len(val_labels)}"
                    assert not np.isnan(cal_probs).any(), f"{method} produced NaN probabilities; check inputs/training"

                    save_calibrated_probs_to_txt(val_pairs, val_labels, val_logits, cal_probs,
                                               fold_num, model_type, method, "val", output_dir)

                    print(f"\n{'='*70}")
                    print(f"📊 Validation evaluation: {method.upper()} (Fold {fold_num}, {model_type.upper()})")
                    print(f"{'='*70}")
                    results = evaluate_calibration(val_labels, cal_probs, method, fold_num, output_dir, 
                                                  optimal_threshold=None, dataset_type='val')
                    results['fold'] = fold_num
                    results['dataset'] = 'val'
                    val_optimal_threshold = results['optimal_threshold']
                    kfold_results[model_type].append(results)

                    cal_path = os.path.join(output_dir, f"calibrator_{method}_fold{fold_num}.pkl")
                    calibrator.save(cal_path, method=method, model_type=model_type, fold=fold_num)
                    print(f"\n✅ Saved calibrator: {cal_path}")
                    
                    threshold_file = os.path.join(output_dir, f"optimal_threshold_{method}_fold{fold_num}.txt")
                    with open(threshold_file, 'w') as f:
                        f.write(f"Method: {method}\n")
                        f.write(f"Fold: {fold_num}\n")
                        f.write(f"Model Type: {model_type}\n")
                        f.write(f"Optimal Threshold (from validation set): {val_optimal_threshold:.6f}\n")
                        f.write(f"Validation Precision: {results.get('opt_precision', 'N/A')}\n")
                        f.write(f"Validation Recall: {results.get('opt_recall', 'N/A')}\n")
                    print(f"✅ Saved optimal threshold: {threshold_file}")

                    print(f"\n{'='*70}")
                    print(f"🧪 Test evaluation: {method.upper()} (Fold {fold_num}, {model_type.upper()})")
                    print(f"🔑 Using validation optimal threshold: {val_optimal_threshold:.4f}")
                    print(f"{'='*70}")
                    test_logits, test_labels, test_pairs = get_logits_from_model_batch(model, test_data, batch_size=batch_size, cache_dir=lanjz_dir, model_type=model_type, model_key=f"fold{fold_num}_test")

                    if len(test_logits) > 0:
                        if not os.path.exists(os.path.join(lanjz_dir, f"wet_experiment_logits_cache_{get_data_hash(test_data)}_{model_type}_fold{fold_num}_test.pkl")):
                            save_logits_cache(get_data_hash(test_data), model_type, test_logits, test_labels, test_pairs, lanjz_dir, model_key=f"fold{fold_num}_test")
                            print(f"  ✅ Saved test logits cache: fold{fold_num}_test")
                        
                        # Apply calibration to the test set
                        test_cal_probs = calibrator.predict_proba(test_logits)

                        # Save test calibrated results
                        test_output_dir = os.path.join(ceshi_jiaozhun_dir, f"calibrated_models_{model_type}")
                        os.makedirs(test_output_dir, exist_ok=True)
                        save_calibrated_probs_to_txt(test_pairs, test_labels, test_logits, test_cal_probs,
                                                   fold_num, model_type, method, "test", test_output_dir)

                        # Evaluate test set using validation optimal threshold (no re-optimization)
                        test_results = evaluate_calibration(test_labels, test_cal_probs, method, fold_num, test_output_dir, 
                                                          optimal_threshold=val_optimal_threshold, dataset_type='test')
                        test_results['fold'] = fold_num
                        test_results['dataset'] = 'test'
                        test_results['threshold_source'] = 'validation'  # mark threshold source

                        # Save test results
                        pingjia_output_dir = os.path.join(fold_pingjia_dir, f"calibrated_models_{model_type}")
                        os.makedirs(pingjia_output_dir, exist_ok=True)
                        test_results_file = os.path.join(pingjia_output_dir, f"test_results_{method}_fold{fold_num}.json")
                        with open(test_results_file, 'w') as f:
                            json.dump(test_results, f, indent=2)
                        print(f"\n✅ Saved test metrics: {test_results_file}")
                        print(f"   🔑 Threshold source: validation ({val_optimal_threshold:.4f})")
                        print(f"   📊 Precision: {test_results['precision']:.4f}, Recall: {test_results['recall']:.4f}")

                        # Save validation results
                        val_results_file = os.path.join(pingjia_output_dir, f"val_results_{method}_fold{fold_num}.json")
                        with open(val_results_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        print(f"✅ Saved validation metrics: {val_results_file}")
                        print(f"   🎯 Optimized threshold: {val_optimal_threshold:.4f}")
                        print(f"   📊 Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}")

                        # Save calibration method parameters
                        threshold_txt = os.path.join(fold_pingjia_dir, "threshold_parameters_summary.txt")
                        
                        # Get calibrator parameters
                        cal_params = calibrator.get_params()
                        
                        # Check if threshold fell back to 0.5
                        val_threshold_fallback = results.get('threshold_fallback', False)
                        
                        with open(threshold_txt, 'a') as f:
                            # Validation info
                            f.write(f"Calibration method: {method}\n")
                            f.write(f"Fold: {fold_num}\n")
                            f.write(f"Model Type: {model_type}\n")
                            f.write("Dataset: val\n")
                            f.write(f"Optimal threshold: {val_optimal_threshold:.4f}\n")
                            
                            # Add warning if threshold fell back to 0.5
                            if val_threshold_fallback and abs(val_optimal_threshold - 0.5) < 0.0001:
                                f.write("⚠️ Note: no threshold met precision >= 0.82; fell back to default 0.5\n")
                            
                            f.write("Parameters:\n")
                            if cal_params:
                                for param_name, param_value in cal_params.items():
                                    if isinstance(param_value, (list, np.ndarray)):
                                        f.write(f"  {param_name}: [array with {len(param_value)} points]\n")
                                    else:
                                        f.write(f"  {param_name}: {param_value:.6f}\n")
                            else:
                                f.write("  No parameter info\n")
                            f.write("-" * 50 + "\n")
                            
                            # Test info
                            f.write("Dataset: test\n")
                            f.write(f"Threshold used: {val_optimal_threshold:.4f} (from validation)\n")
                            
                            # Note threshold source for test as well
                            if val_threshold_fallback and abs(val_optimal_threshold - 0.5) < 0.0001:
                                f.write("⚠️ Note: test threshold comes from validation default 0.5 (no valid threshold found)\n")
                            
                            f.write("Parameters:\n")
                            if cal_params:
                                for param_name, param_value in cal_params.items():
                                    if isinstance(param_value, (list, np.ndarray)):
                                        f.write(f"  {param_name}: [array with {len(param_value)} points]\n")
                                    else:
                                        f.write(f"  {param_name}: {param_value:.6f}\n")
                            else:
                                f.write("  No parameter info\n")
                            f.write("=" * 50 + "\n\n")


        # Save 5-fold CV summary
        for model_type in model_types:
            if kfold_results[model_type]:
                results_df = pd.DataFrame(kfold_results[model_type])
                output_dir = os.path.join(fold_jiaozhun_dir, f"calibrated_models_{model_type}")
                results_df.to_csv(os.path.join(output_dir, "kfold_calibration_results.csv"), index=False)

                print(f"\n=== {model_type.upper()} 5-fold CV summary ===")
                summary = results_df.groupby('method').agg({
                    'ece': ['mean', 'std'],
                    'brier': ['mean', 'std'],
                    'auroc': ['mean', 'std'],
                    'auprc': ['mean', 'std']
                }).round(4)
                print(summary)

        # Aggregate JSON metrics into CSV report
        print("\nAggregating JSON metrics to CSV report...")
        all_results = []
        
        for model_type in model_types:
            pingjia_dir = os.path.join(fold_pingjia_dir, f"calibrated_models_{model_type}")
            if not os.path.exists(pingjia_dir):
                continue
                
            # Iterate all JSON files
            for json_file in os.listdir(pingjia_dir):
                if json_file.endswith('.json'):
                    json_path = os.path.join(pingjia_dir, json_file)
                    try:
                        with open(json_path, 'r') as f:
                            result = json.load(f)
                            result['model_type'] = model_type
                            all_results.append(result)
                    except Exception as e:
                        print(f"Failed to read JSON file {json_path}: {e}")
        
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_csv = os.path.join(fold_pingjia_dir, "all_metrics_summary.csv")
            summary_df.to_csv(summary_csv, index=False)
            print(f"Saved aggregated metrics to: {summary_csv}")
            
            print("\n=== Aggregated metrics summary ===")
            if 'dataset' in summary_df.columns:
                grouped_stats = summary_df.groupby(['method', 'dataset']).agg({
                    'ece': ['mean', 'std', 'min', 'max'],
                    'brier': ['mean', 'std', 'min', 'max'],
                    'auroc': ['mean', 'std', 'min', 'max'],
                    'auprc': ['mean', 'std', 'min', 'max'],
                    'accuracy': ['mean', 'std']
                }).round(4)
                print(grouped_stats)

    # 2. Final calibrator (use all data)
    if do_final and use_wet_experiment_data:
        print("\n" + "="*80)
        print("Starting final calibrator (using all data)")
        print("="*80)

        for fold_num in folds:
            print(f"\nProcessing final calibrator - Fold {fold_num} model")

            for model_type in model_types:
                print(f"\n--- Processing {model_type.upper()} model ---")

                output_dir = os.path.join(final_jiaozhun_dir, f"calibrated_models_{model_type}")
                os.makedirs(output_dir, exist_ok=True)

                model_path = os.path.join(base_dir, f"fold{fold_num}_best_{model_type}.pt")
                if not os.path.exists(model_path):
                    print(f"Model file not found: {model_path}")
                    continue

                checkpoint = torch.load(model_path, map_location='cpu')
                actual_in_dim = checkpoint['cfc_preprocess.cells.0.backbone.0.weight'].shape[1]

                model = None
                for try_in_dim in [actual_in_dim + 1, actual_in_dim, actual_in_dim - 1, actual_in_dim - 2]:
                    try:
                        model = MyGCN(in_dim=try_in_dim, nhid=256, dropout=0.4, time_steps=5)
                        model.load_state_dict(checkpoint, strict=False)
                        print(f"Loaded model successfully with in_dim={try_in_dim}")
                        break
                    except RuntimeError as e:
                        continue
                else:
                    print(f"Failed to load model, skipping {model_type.upper()}")
                    continue

                model.eval()

                print(f"Computing logits for all data with {len(train_val_data) + len(test_data)} samples")
                all_data = train_val_data + test_data
                all_logits, all_labels, all_pairs = get_logits_from_model_batch(model, all_data, batch_size=batch_size, cache_dir=lanjz_dir, model_type=model_type, model_key=f"fold{fold_num}_final")

                if len(all_logits) == 0:
                    print("No valid logits, skipping")
                    continue

                # Save logits cache for all data (if not already cached)
                if not os.path.exists(os.path.join(lanjz_dir, f"wet_experiment_logits_cache_{get_data_hash(all_data)}_{model_type}_fold{fold_num}_final.pkl")):
                    save_logits_cache(get_data_hash(all_data), model_type, all_logits, all_labels, all_pairs, lanjz_dir, model_key=f"fold{fold_num}_final")
                    print(f"Saved all-data logits cache: fold{fold_num}_final")

                # Save logits for all data
                save_logits_to_txt(all_pairs, all_labels, all_logits, fold_num, model_type, output_dir)

                # For each calibration method
                for method in methods:
                    print(f"\n{'='*60}")
                    print(f"Final calibrator - method: {method.upper()}")
                    print(f"{'='*60}")

                    # Create calibrator (tune hyperparameters with inner CV on all data)
                    if method == 'platt':
                        # Tune Platt params with inner CV on all data
                        best_params = tune_platt_scaling_cv(all_logits, all_labels, cv_folds=3)
                        calibrator = PlattScaling(**best_params)
                    elif method == 'beta':
                        # Tune Beta params with inner CV on all data
                        best_params = tune_beta_calibration_cv(all_logits, all_labels, cv_folds=3)
                        calibrator = BetaCalibration(**best_params)
                    elif method == 'temperature':
                        # Temperature scaling is determined by optimization
                        tune_temperature_scaling_cv(all_logits, all_labels, cv_folds=3)
                        calibrator = TemperatureScaling()
                    elif method == 'isotonic':
                        # Tune Isotonic params with inner CV on all data
                        best_params = tune_isotonic_calibration_cv(all_logits, all_labels, cv_folds=3)
                        calibrator = IsotonicCalibration(**best_params)
                    elif method == 'boosting':
                        # Tune Boosting params with inner CV on all data
                        best_params = tune_boosting_calibration_cv(all_logits, all_labels, cv_folds=3)
                        calibrator = BoostingCalibration(**best_params)
                    else:
                        raise ValueError(f"Unknown calibration method: {method}")

                    print(f"Method implementation: {method} -> {type(calibrator).__name__}")

                    # Fit calibrator (using all data)
                    calibrator.fit(all_logits, all_labels)

                    # Apply calibration to all data
                    cal_probs = calibrator.predict_proba(all_logits)

                    # Basic consistency checks
                    assert len(cal_probs) == len(all_labels), f"{method} output length mismatch: {len(cal_probs)} vs {len(all_labels)}"
                    assert not np.isnan(cal_probs).any(), f"{method} produced NaN probabilities; check inputs/training"

                    # Save calibrated results
                    save_calibrated_probs_to_txt(all_pairs, all_labels, all_logits, cal_probs,
                                               fold_num, model_type, method, "final", output_dir)

                    # Evaluate calibration (final calibrator uses all data and optimizes threshold on it)
                    print(f"\n{'='*70}")
                    print(f"📊 Final calibrator evaluation: {method.upper()} (using all train+val data, {model_type.upper()})")
                    print(f"{'='*70}")
                    results = evaluate_calibration(all_labels, cal_probs, method, fold_num, output_dir, 
                                                  optimal_threshold=None, dataset_type='val')
                    results['fold'] = fold_num
                    results['dataset'] = 'final'
                    final_results[model_type].append(results)

                    # Save calibrator
                    cal_path = os.path.join(output_dir, f"final_calibrator_{method}_fold{fold_num}.pkl")
                    calibrator.save(cal_path, method=method, model_type=model_type, fold=fold_num)
                    print(f"Saved final calibrator: {cal_path}")

        # Save final calibrator summary
        for model_type in model_types:
            if final_results[model_type]:
                results_df = pd.DataFrame(final_results[model_type])
                output_dir = os.path.join(final_jiaozhun_dir, f"calibrated_models_{model_type}")
                results_df.to_csv(os.path.join(output_dir, "final_calibration_results.csv"), index=False)

                print(f"\n=== {model_type.upper()} final calibrator summary ===")
                summary = results_df.groupby('method').agg({
                    'ece': ['mean', 'std'],
                    'brier': ['mean', 'std'],
                    'auroc': ['mean', 'std'],
                    'auprc': ['mean', 'std']
                }).round(4)
                print(summary)

    print("All calibration tasks completed!")
    print(f"5-fold data saved to: {shuju_fold_dir}")
    print(f"5-fold calibration results saved to: {fold_jiaozhun_dir}")
    print(f"5-fold metrics saved to: {fold_pingjia_dir}")
    print(f"Test calibration results saved to: {ceshi_jiaozhun_dir}")
    print(f"Final calibrator results saved to: {final_jiaozhun_dir}")
    print(f"Lazy logits cache saved to: {lanjz_dir}")

if __name__ == "__main__":
    model_types = ['auprc', 'auroc', 'f1']  # model types to calibrate
    folds = list(range(1, 6))  # process all 5 folds
    batch_size = 64  # batch size
    use_lazy_loading = True  # use lazy loading by default
    cache_dir = None  # cache dir, defaults to output_base_dir/logits_cache

    # Data config
    use_wet_experiment_data = True
    wet_pos_file = "/teams/YingChiLab_1702378116/YuemingXiao/PPI_yuece_new/jiaozhun/3000_pos.txt"
    wet_neg_file = "/teams/YingChiLab_1702378116/YuemingXiao/PPI_yuece_new/jiaozhun/3000_neg_qf.txt"

    # Calibration task config
    do_kfold = True   # run 5-fold cross-validation
    do_final = False   # run final calibrator

    print("Starting PPI model calibration...")
    print(f"Model types: {model_types}")
    print(f"Folds: {folds}")
    print(f"Batch size: {batch_size}")
    print(f"Use lazy loading: {use_lazy_loading}")
    print(f"Use wet-lab data: {use_wet_experiment_data}")
    print(f"Run 5-fold cross-validation: {do_kfold}")
    print(f"Run final calibrator: {do_final}")
    if use_wet_experiment_data:
        print(f"Positive file: {wet_pos_file}")
        print(f"Negative file: {wet_neg_file}")
    print("-" * 50)

    calibrate_and_save_models_kfold(
        model_types=model_types,
        folds=folds,
        batch_size=batch_size,
        use_lazy_loading=use_lazy_loading,
        cache_dir=cache_dir,
        use_wet_experiment_data=use_wet_experiment_data,
        wet_pos_file=wet_pos_file,
        wet_neg_file=wet_neg_file,
        do_kfold=do_kfold,
        do_final=do_final
    )