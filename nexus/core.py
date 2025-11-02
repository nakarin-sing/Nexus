#!/usr/bin/env python3
"""
NEXUS Core v4.0.1 — Production Release
Neural EXpert Unified System for Online Learning

Fixed bugs:
- Bernoulli return type → Dict[bool, float]
- Added input validation
- Optimized snapshot similarity calculation
- Added comprehensive error handling

MIT License | 2025
"""

from __future__ import annotations

import numpy as np
import logging
from collections import deque
from typing import Dict, Any, Optional, Final, Literal, List
from dataclasses import dataclass
from pathlib import Path
import pickle
from threading import RLock
from typing_extensions import Self
import warnings

from river.base import Classifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== CONSTANTS ====================
STRESS_HIGH: Final[float] = 0.15
STRESS_MED: Final[float] = 0.05
LOSS_HIGH_THRESH: Final[float] = 0.5
LOSS_MED_THRESH: Final[float] = 0.3
LR_MIN: Final[float] = 0.01
LR_MAX: Final[float] = 1.0
SIM_THRESH: Final[float] = 0.85
EPS: Final[float] = 1e-9
STD_EPS: Final[float] = 1e-6
GRAD_CLIP: Final[float] = 1.0
MIN_WEIGHT: Final[float] = 0.1
NCRA_MIN_SIM: Final[float] = 0.1
WEIGHT_DECAY: Final[float] = 0.9995
NUMPY_FLOAT: Final[type] = np.float32

# Ensemble weights
MAIN_WEIGHT: Final[float] = 1.0
NCRA_WEIGHT: Final[float] = 0.7
RFC_WEIGHT: Final[float] = 0.5

# ==================== CONFIG ====================
@dataclass(frozen=True)
class Config:
    """Global configuration for NEXUS"""
    seed: int = 42
    max_snapshots: int = 5
    stress_history_len: int = 1000
    version: str = "4.0.1"
    weight_decay: float = WEIGHT_DECAY

CONFIG = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("NEXUS")

# ==================== UTILITIES ====================
def safe_div(a: float, b: float) -> float:
    """Safe division with epsilon"""
    return a / (b + EPS)

def safe_exp(x: float) -> float:
    """Safe exponential with clipping"""
    return np.exp(np.clip(x, -20.0, 20.0))

def safe_std(arr: np.ndarray) -> float:
    """Safe standard deviation"""
    return max(float(np.std(arr, ddof=0)), STD_EPS)

def safe_norm(arr: np.ndarray) -> float:
    """Safe L2 norm"""
    norm = float(np.linalg.norm(arr))
    return norm if norm > 0 else EPS

# ==================== SNAPSHOT ====================
@dataclass
class Snapshot:
    """Memory snapshot with cached norm for efficiency"""
    w: np.ndarray
    bias: float
    context: np.ndarray
    weight: float
    context_norm: float
    
    @classmethod
    def create(cls, w: np.ndarray, bias: float, context: np.ndarray, weight: float = 1.0) -> Self:
        """Factory method with automatic norm calculation"""
        return cls(
            w=w.copy(),
            bias=bias,
            context=context.copy(),
            weight=weight,
            context_norm=safe_norm(context)
        )

# ==================== NEXUS CORE ====================
class NEXUS_River(Classifier):
    """NEXUS: Neural EXpert Unified System
    
    A memory-aware online learning algorithm combining:
    - Main logistic regression with adaptive learning rate
    - NCRA: Neural Context-Reactive Adaptation (snapshot ensemble)
    - RFC: Residual Feedback Correction (error correction)
    
    Parameters
    ----------
    dim : int, optional
        Feature dimensionality. Auto-detected if None.
    enable_ncra : bool, default=True
        Enable Neural Context-Reactive Adaptation.
    enable_rfc : bool, default=True
        Enable Residual Feedback Correction.
    max_snapshots : int, default=5
        Maximum number of snapshots to store.
    
    Attributes
    ----------
    w : np.ndarray
        Main model weights
    stress : float
        Current model stress level (0-1)
    snapshots : deque[Snapshot]
        Stored model snapshots for NCRA
    sample_count : int
        Number of samples processed
    
    Examples
    --------
    >>> from river import datasets, metrics
    >>> model = NEXUS_River()
    >>> metric = metrics.ROCAUC()
    >>> 
    >>> for x, y in datasets.Phishing().take(1000):
    ...     y_pred = model.predict_proba_one(x)
    ...     model.learn_one(x, y)
    ...     metric.update(y, y_pred[True])
    >>> 
    >>> print(f"AUC: {metric.get():.4f}")
    
    Notes
    -----
    - Thread-safe with RLock
    - Memory-safe with bounded snapshot buffer
    - Handles dynamic feature spaces
    """

    def __init__(
        self, 
        dim: Optional[int] = None, 
        enable_ncra: bool = True, 
        enable_rfc: bool = True,
        max_snapshots: int = CONFIG.max_snapshots
    ):
        super().__init__()
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive")
        
        self.dim: Optional[int] = dim
        self.w: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.lr: float = 0.08
        self.stress: float = 0.0
        self.stress_history: deque[float] = deque(maxlen=CONFIG.stress_history_len)
        self.snapshots: deque[Snapshot] = deque(maxlen=max_snapshots)
        self.rfc_w: Optional[np.ndarray] = None
        self.rfc_bias: float = 0.0
        self.rfc_lr: float = 0.01
        self.sample_count: int = 0
        self.feature_names: List[str] = []
        self.enable_ncra: bool = enable_ncra
        self.enable_rfc: bool = enable_rfc
        self._lock: RLock = RLock()

    def _init_weights(self, n_features: int) -> None:
        """Initialize model weights"""
        if self.dim is None:
            self.dim = n_features
        scale = 0.1 / np.sqrt(self.dim)
        self.w = np.random.normal(0, scale, self.dim).astype(NUMPY_FLOAT)
        if self.enable_rfc:
            self.rfc_w = np.random.normal(0, scale, self.dim).astype(NUMPY_FLOAT)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation"""
        return 1.0 / (1.0 + safe_exp(-x))

    def _validate_features(self, x: Dict[str, Any]) -> None:
        """Validate feature dictionary"""
        if not x:
            raise ValueError("Feature dictionary cannot be empty")
        
        for key in x:
            if not isinstance(key, str):
                raise TypeError(f"Feature keys must be strings, got {type(key)}")
            if len(key) > 1000:
                raise ValueError(f"Feature key too long: {len(key)} chars")
        
        for key, val in x.items():
            try:
                float(val)
            except (TypeError, ValueError):
                raise TypeError(f"Feature '{key}' has non-numeric value: {val}")

    def _to_array(self, x: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict to array with dynamic feature handling"""
        if not isinstance(x, dict):
            raise TypeError("x must be a dictionary of features")

        self._validate_features(x)
        
        current_features = set(x.keys())
        if not self.feature_names:
            self.feature_names = sorted(current_features)
            if self.w is not None:
                self._extend_weights(len(self.feature_names))
        else:
            new_features = current_features - set(self.feature_names)
            if new_features:
                self.feature_names.extend(sorted(new_features))
                self._extend_weights(len(self.feature_names))

        arr = np.array(
            [float(x.get(k, 0.0)) for k in self.feature_names], 
            dtype=NUMPY_FLOAT
        )
        arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)
        arr = np.clip(arr, -100.0, 100.0)
        
        if len(arr) > self.dim:
            arr = arr[:self.dim]
        
        return arr

    def _extend_weights(self, new_dim: int) -> None:
        """Extend weight vectors for new features"""
        if self.w is None:
            return
        old_dim = len(self.w)
        if new_dim > old_dim:
            pad = np.zeros(new_dim - old_dim, dtype=NUMPY_FLOAT)
            self.w = np.concatenate([self.w, pad])
            if self.rfc_w is not None:
                self.rfc_w = np.concatenate([self.rfc_w, pad])
            self.dim = new_dim

    def _get_context(self, x_arr: np.ndarray) -> np.ndarray:
        """Extract context features from input"""
        std = safe_std(x_arr)
        return np.array([std, self.stress], dtype=NUMPY_FLOAT)

    def predict_one(self, x: Dict[str, Any]) -> Literal[0, 1]:
        """Predict binary class label
        
        Parameters
        ----------
        x : dict
            Feature dictionary
            
        Returns
        -------
        int
            Predicted class (0 or 1)
        """
        proba = self.predict_proba_one(x)
        return 1 if proba[True] >= 0.5 else 0

    def predict_proba_one(self, x: Dict[str, Any]) -> Dict[bool, float]:
        """Predict class probabilities
        
        Parameters
        ----------
        x : dict
            Feature dictionary
            
        Returns
        -------
        dict
            Probability distribution {True: p, False: 1-p}
        """
        with self._lock:
            if self.w is None:
                self._init_weights(len(x))
            
            x_arr = self._to_array(x)
            
            # Main prediction
            p_main = self._sigmoid(np.dot(x_arr, self.w) + self.bias)
            
            # NCRA prediction
            p_ncra = self._predict_ncra(x_arr) if (
                self.enable_ncra and self.snapshots
            ) else p_main
            
            # RFC prediction
            p_rfc = self._sigmoid(
                np.dot(x_arr, self.rfc_w) + self.rfc_bias
            ) if (self.enable_rfc and self.rfc_w is not None) else p_main

            # Ensemble
            w_m = MAIN_WEIGHT
            w_n = NCRA_WEIGHT if (self.enable_ncra and self.snapshots) else 0.0
            w_r = RFC_WEIGHT if self.enable_rfc else 0.0
            total = w_m + w_n + w_r + EPS
            
            p_ens = safe_div(w_m * p_main + w_n * p_ncra + w_r * p_rfc, total)
            p_ens = float(np.clip(p_ens, 0.0, 1.0))
            
            return {True: p_ens, False: 1.0 - p_ens}

    def _predict_ncra(self, x: np.ndarray) -> float:
        """NCRA ensemble prediction from snapshots"""
        if not self.snapshots:
            return 0.5
        
        context = self._get_context(x)
        context_norm = safe_norm(context)
        
        preds: List[float] = []
        weights: List[float] = []
        
        for snapshot in self.snapshots:
            sim = np.dot(context, snapshot.context) / (
                context_norm * snapshot.context_norm
            )
            if sim < NCRA_MIN_SIM:
                continue
            
            logit = np.dot(x, snapshot.w) + float(snapshot.bias)
            preds.append(self._sigmoid(logit))
            weights.append(float(snapshot.weight) * max(0.0, sim))
        
        if not weights:
            return 0.5
        
        total = sum(weights) + EPS
        return float(np.average(preds, weights=[w / total for w in weights]))

    def learn_one(self, x: Dict[str, Any], y: Literal[0, 1]) -> Self:
        """Update model with one sample
        
        Parameters
        ----------
        x : dict
            Feature dictionary
        y : int
            True label (0 or 1)
            
        Returns
        -------
        self
        """
        if y not in {0, 1}:
            raise ValueError("y must be 0 or 1")

        with self._lock:
            self.sample_count += 1
            
            if self.w is None:
                self._init_weights(len(x))
            
            x_arr = self._to_array(x)

            # Get predictions
            p_main = self._sigmoid(np.dot(x_arr, self.w) + self.bias)
            proba = self.predict_proba_one(x)
            p_ens = proba[True]
            err = p_ens - float(y)

            # Adaptive learning rate
            adaptive_lr = np.clip(
                self.lr * (1.0 + min(self.stress * 3.0, 5.0)), 
                LR_MIN, 
                LR_MAX
            )
            
            # Update main model
            grad = np.clip(adaptive_lr * err * x_arr, -GRAD_CLIP, GRAD_CLIP)
            self.w = (self.w - grad).astype(NUMPY_FLOAT)
            self.bias -= adaptive_lr * err

            # Update RFC
            if self.enable_rfc and self.rfc_w is not None:
                self.rfc_w = (
                    self.rfc_w - self.rfc_lr * (p_main - y) * x_arr
                ).astype(NUMPY_FLOAT)
                self.rfc_bias -= self.rfc_lr * (p_main - y)

            # Update stress
            loss = err ** 2
            new_stress = (
                STRESS_HIGH if loss > LOSS_HIGH_THRESH 
                else STRESS_MED if loss > LOSS_MED_THRESH 
                else 0.0
            )
            self.stress = 0.9 * self.stress + 0.1 * new_stress
            self.stress_history.append(self.stress)

            # NCRA: Snapshot management
            if self.enable_ncra:
                stress_thresh = float(
                    np.percentile(list(self.stress_history)[-100:], 80)
                ) if len(self.stress_history) > 100 else STRESS_HIGH
                
                context = self._get_context(x_arr)
                
                # Check similarity with existing snapshots
                if self.snapshots:
                    context_norm = safe_norm(context)
                    sims = [
                        np.dot(context, s.context) / (context_norm * s.context_norm)
                        for s in self.snapshots
                    ]
                    if max(sims) > SIM_THRESH:
                        return self
                
                # Create new snapshot if stressed
                if self.stress > stress_thresh:
                    self.snapshots.append(
                        Snapshot.create(self.w, self.bias, context)
                    )
                
                # Update snapshot weights
                if self.snapshots:
                    err_ncra = abs(self._predict_ncra(x_arr) - y)
                    context_norm = safe_norm(context)
                    
                    for snapshot in self.snapshots:
                        sim = np.dot(context, snapshot.context) / (
                            context_norm * snapshot.context_norm
                        )
                        snapshot.weight = max(
                            MIN_WEIGHT,
                            snapshot.weight * safe_exp(-5 * err_ncra) * (1 + 0.5 * max(0, sim))
                        )
                        if snapshot.weight > MIN_WEIGHT * 2:
                            snapshot.weight *= CONFIG.weight_decay
                    
                    # Normalize weights
                    total = sum(s.weight for s in self.snapshots) + EPS
                    for snapshot in self.snapshots:
                        snapshot.weight /= total

            return self

    def reset(self) -> None:
        """Reset internal state for reuse"""
        with self._lock:
            self.sample_count = 0
            self.stress = 0.0
            self.stress_history.clear()
            self.snapshots.clear()
            self.feature_names = []

    def save(self, path: str) -> None:
        """Save model state to file
        
        Parameters
        ----------
        path : str
            File path to save model
        """
        with self._lock:
            state = {k: v for k, v in self.__dict__.items() if k != "_lock"}
            with open(path, 'wb') as f:
                pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load model from file
        
        Parameters
        ----------
        path : str
            File path to load model from
            
        Returns
        -------
        NEXUS_River
            Loaded model instance
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(
            dim=state["dim"], 
            enable_ncra=state["enable_ncra"], 
            enable_rfc=state["enable_rfc"]
        )
        model.__dict__.update(state)
        model._lock = RLock()
        model.feature_names = state.get("feature_names", [])
        return model

    def __repr__(self) -> str:
        return (
            f"NEXUS_River(v{CONFIG.version}, dim={self.dim}, "
            f"samples={self.sample_count}, snapshots={len(self.snapshots)})"
        )
