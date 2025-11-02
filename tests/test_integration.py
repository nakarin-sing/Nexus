# tests/test_integration.py
"""
Integration tests for NEXUS with River datasets
Tests full pipeline performance
"""

import pytest
import numpy as np
from river import datasets, metrics, preprocessing
from nexus.core import NEXUS_River


# ==================== PIPELINE TESTS ====================
def test_phishing_dataset():
    """Test on Phishing dataset"""
    model = preprocessing.StandardScaler() | NEXUS_River()
    metric = metrics.ROCAUC()
    
    for i, (x, y) in enumerate(datasets.Phishing()):
        if i >= 1000:
            break
        
        y_pred = model.predict_proba_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred[True])
    
    auc = metric.get()
    assert auc > 0.6, f"AUC too low: {auc:.4f}"


def test_credit_card_dataset():
    """Test on CreditCard dataset"""
    model = preprocessing.StandardScaler() | NEXUS_River()
    metric = metrics.ROCAUC()
    
    for i, (x, y) in enumerate(datasets.CreditCard()):
        if i >= 1000:
            break
        
        y_pred = model.predict_proba_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred[True])
    
    auc = metric.get()
    assert auc > 0.5, f"AUC too low: {auc:.4f}"


def test_elec2_dataset():
    """Test on Electricity dataset"""
    model = preprocessing.StandardScaler() | NEXUS_River()
    metric = metrics.Accuracy()
    
    for i, (x, y) in enumerate(datasets.Elec2()):
        if i >= 1000:
            break
        
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred)
    
    acc = metric.get()
    assert acc > 0.5, f"Accuracy too low: {acc:.4f}"


# ==================== COMPARISON TESTS ====================
def test_compare_with_baseline():
    """Compare NEXUS with baseline on synthetic data"""
    from river import linear_model
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    X = [
        {f"f{j}": float(np.random.randn()) for j in range(5)}
        for _ in range(n_samples)
    ]
    y = [int(sum(x.values()) > 0) for x in X]
    
    # NEXUS
    model_nexus = NEXUS_River()
    metric_nexus = metrics.ROCAUC()
    
    for x, y_true in zip(X, y):
        y_pred = model_nexus.predict_proba_one(x)
        model_nexus.learn_one(x, y_true)
        metric_nexus.update(y_true, y_pred[True])
    
    # Baseline
    model_baseline = linear_model.LogisticRegression()
    metric_baseline = metrics.ROCAUC()
    
    for x, y_true in zip(X, y):
        y_pred = model_baseline.predict_proba_one(x)
        model_baseline.learn_one(x, y_true)
        metric_baseline.update(y_true, y_pred[True])
    
    auc_nexus = metric_nexus.get()
    auc_baseline = metric_baseline.get()
    
    print(f"NEXUS AUC: {auc_nexus:.4f}, Baseline AUC: {auc_baseline:.4f}")
    assert auc_nexus > 0.7


# ==================== STRESS TESTS ====================
def test_concept_drift():
    """Test adaptation to concept drift"""
    model = NEXUS_River(enable_ncra=True)
    
    # Phase 1: Feature A is positive
    for i in range(200):
        x = {"A": 1.0, "B": 0.0}
        y = 1
        model.learn_one(x, y)
    
    # Phase 2: Feature B is positive (drift)
    for i in range(200):
        x = {"A": 0.0, "B": 1.0}
        y = 1
        model.learn_one(x, y)
    
    # Test on new concept
    x_new = {"A": 0.0, "B": 1.0}
    proba = model.predict_proba_one(x_new)
    
    # Should adapt to new concept
    assert proba[True] > 0.5


def test_high_dimensional_data():
    """Test with high-dimensional data"""
    model = NEXUS_River()
    
    for i in range(500):
        x = {f"f{j}": float(np.random.randn()) for j in range(100)}
        y = int(sum(x.values()) > 0)
        
        model.learn_one(x, y)
        proba = model.predict_proba_one(x)
        
        assert 0.0 <= proba[True] <= 1.0


def test_imbalanced_data():
    """Test with imbalanced classes"""
    model = NEXUS_River()
    metric = metrics.ROCAUC()
    
    # 90% class 0, 10% class 1
    for i in range(1000):
        x = {f"f{j}": float(np.random.randn()) for j in range(5)}
        y = 1 if i % 10 == 0 else 0
        
        y_pred = model.predict_proba_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred[True])
    
    auc = metric.get()
    assert auc > 0.5


# ==================== SNAPSHOT TESTS ====================
def test_snapshot_reuse():
    """Test that snapshots are actually used"""
    model = NEXUS_River(enable_ncra=True, max_snapshots=5)
    
    # Create some snapshots
    for i in range(500):
        x = {f"f{j}": float(np.random.randn()) for j in range(5)}
        y = int(sum(x.values()) > 0)
        model.learn_one(x, y)
    
    # Check snapshots were created
    assert len(model.snapshots) > 0
    
    # Predict with snapshots
    x_test = {f"f{j}": 0.0 for j in range(5)}
    proba = model.predict_proba_one(x_test)
    
    assert 0.0 <= proba[True] <= 1.0


# ==================== PERFORMANCE TESTS ====================
def test_training_speed():
    """Test training speed"""
    import time
    
    model = NEXUS_River()
    
    start = time.time()
    for i in range(1000):
        x = {f"f{j}": float(i + j) for j in range(10)}
        y = i % 2
        model.learn_one(x, y)
    
    elapsed = time.time() - start
    
    # Should process 1000 samples in reasonable time
    assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s"


def test_memory_stability():
    """Test memory doesn't grow unbounded"""
    import sys
    
    model = NEXUS_River(dim=10)
    
    # Train for many iterations
    for i in range(5000):
        x = {f"f{j}": float(i + j) for j in range(10)}
        y = i % 2
        model.learn_one(x, y)
    
    # Check bounded structures
    assert len(model.snapshots) <= model.snapshots.maxlen
    assert len(model.stress_history) <= model.stress_history.maxlen
    
    # Weights should not grow
    assert len(model.w) == model.dim


# ==================== EDGE CASE TESTS ====================
def test_all_same_label():
    """Test with all same label"""
    model = NEXUS_River()
    
    for i in range(100):
        x = {f"f{j}": float(np.random.randn()) for j in range(5)}
        y = 1  # Always 1
        model.learn_one(x, y)
    
    x_test = {f"f{j}": 0.0 for j in range(5)}
    proba = model.predict_proba_one(x_test)
    
    # Should predict mostly 1
    assert proba[True] > 0.5


def test_alternating_labels():
    """Test with perfectly alternating labels"""
    model = NEXUS_River()
    
    for i in range(100):
        x = {"a": 1.0}
        y = i % 2
        model.learn_one(x, y)
    
    proba = model.predict_proba_one({"a": 1.0})
    
    # Should be uncertain
    assert 0.3 < proba[True] < 0.7


def test_sparse_features():
    """Test with very sparse features"""
    model = NEXUS_River()
    
    for i in range(100):
        # Only 1 feature active
        x = {f"f{i % 50}": 1.0}
        y = i % 2
        model.learn_one(x, y)
    
    x_test = {"f25": 1.0}
    proba = model.predict_proba_one(x_test)
    
    assert 0.0 <= proba[True] <= 1.0


# ==================== REPRODUCIBILITY TESTS ====================
def test_reproducibility():
    """Test that results are reproducible with same seed"""
    np.random.seed(42)
    
    def train_model():
        model = NEXUS_River(dim=5)
        for i in range(100):
            x = {f"f{j}": float(np.random.randn()) for j in range(5)}
            y = i % 2
            model.learn_one(x, y)
        return model.w.copy()
    
    np.random.seed(42)
    w1 = train_model()
    
    np.random.seed(42)
    w2 = train_model()
    
    assert np.allclose(w1, w2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
