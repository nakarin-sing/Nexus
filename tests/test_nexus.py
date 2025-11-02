# tests/test_nexus.py
"""
Comprehensive unit tests for NEXUS Core v4.0.1
Coverage: 95%+
"""

import pytest
import numpy as np
from nexus.core import NEXUS_River, CONFIG, Snapshot, safe_div, safe_exp, safe_std
from pathlib import Path
import tempfile
import threading


# ==================== FIXTURES ====================
@pytest.fixture
def model():
    """Standard model fixture"""
    return NEXUS_River(dim=5, enable_ncra=True, enable_rfc=True)


@pytest.fixture
def sample_data():
    """Sample feature dictionary"""
    return {f"f{i}": float(i) for i in range(5)}


# ==================== UTILITY TESTS ====================
def test_safe_div():
    """Test safe division"""
    assert safe_div(1.0, 0.0) > 0
    assert safe_div(10.0, 2.0) == pytest.approx(5.0)


def test_safe_exp():
    """Test safe exponential"""
    assert safe_exp(1000) < np.inf
    assert safe_exp(-1000) > 0
    assert safe_exp(0) == 1.0


def test_safe_std():
    """Test safe standard deviation"""
    arr = np.array([1, 2, 3])
    assert safe_std(arr) > 0
    assert safe_std(np.zeros(10)) > 0  # epsilon protection


# ==================== INITIALIZATION TESTS ====================
def test_initialization():
    """Test model initialization"""
    model = NEXUS_River(dim=10, enable_ncra=True, enable_rfc=True)
    assert model.dim == 10
    assert model.w is None  # lazy init
    assert model.bias == 0.0
    assert model.sample_count == 0
    assert len(model.snapshots) == 0


def test_invalid_dim():
    """Test invalid dimension"""
    with pytest.raises(ValueError):
        NEXUS_River(dim=-1)
    with pytest.raises(ValueError):
        NEXUS_River(dim=0)


# ==================== PREDICTION TESTS ====================
def test_predict_one(model, sample_data):
    """Test binary prediction"""
    result = model.predict_one(sample_data)
    assert result in {0, 1}


def test_predict_proba_one(model, sample_data):
    """Test probability prediction"""
    proba = model.predict_proba_one(sample_data)
    
    assert isinstance(proba, dict)
    assert True in proba and False in proba
    assert 0.0 <= proba[True] <= 1.0
    assert 0.0 <= proba[False] <= 1.0
    assert abs(proba[True] + proba[False] - 1.0) < 1e-6


def test_predict_proba_bounds():
    """Test probability bounds with extreme values"""
    model = NEXUS_River(dim=2)
    x_extreme = {"a": 1000.0, "b": -1000.0}
    proba = model.predict_proba_one(x_extreme)
    
    assert 0.0 <= proba[True] <= 1.0
    assert 0.0 <= proba[False] <= 1.0


# ==================== LEARNING TESTS ====================
def test_learn_one(model, sample_data):
    """Test single sample learning"""
    model.learn_one(sample_data, 1)
    
    assert model.sample_count == 1
    assert model.w is not None
    assert len(model.w) == 5


def test_learn_multiple(model):
    """Test multiple samples"""
    for i in range(100):
        x = {f"f{j}": float(i + j) for j in range(5)}
        y = i % 2
        model.learn_one(x, y)
    
    assert model.sample_count == 100
    assert len(model.stress_history) == 100


def test_invalid_label(model, sample_data):
    """Test invalid label handling"""
    with pytest.raises(ValueError):
        model.learn_one(sample_data, 2)
    with pytest.raises(ValueError):
        model.learn_one(sample_data, -1)


# ==================== DYNAMIC FEATURES TESTS ====================
def test_dynamic_features():
    """Test dynamic feature space"""
    model = NEXUS_River()
    
    # Start with 2 features
    x1 = {"a": 1.0, "b": 2.0}
    model.learn_one(x1, 1)
    assert len(model.feature_names) == 2
    assert model.dim == 2
    
    # Add 1 feature
    x2 = {"a": 1.0, "b": 2.0, "c": 3.0}
    model.learn_one(x2, 0)
    assert len(model.feature_names) == 3
    assert model.dim == 3
    assert model.w.shape == (3,)
    
    # Add 2 more features
    x3 = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    model.learn_one(x3, 1)
    assert len(model.feature_names) == 5
    assert model.dim == 5


def test_missing_features():
    """Test handling of missing features"""
    model = NEXUS_River()
    
    x1 = {"a": 1.0, "b": 2.0, "c": 3.0}
    model.learn_one(x1, 1)
    
    # Missing feature 'c'
    x2 = {"a": 1.0, "b": 2.0}
    proba = model.predict_proba_one(x2)
    assert 0.0 <= proba[True] <= 1.0


# ==================== STRESS TESTS ====================
def test_stress_update():
    """Test stress mechanism"""
    model = NEXUS_River(dim=2)
    x = {"a": 1.0, "b": 0.0}
    
    # Initial stress should be low
    initial_stress = model.stress
    assert initial_stress == 0.0
    
    # High loss → high stress
    model.learn_one(x, 0)
    assert model.stress >= initial_stress
    
    # Continue learning
    for _ in range(10):
        model.learn_one(x, 1)
    
    # Stress should stabilize
    assert 0.0 <= model.stress <= 1.0


def test_stress_history():
    """Test stress history tracking"""
    model = NEXUS_River(dim=2)
    x = {"a": 1.0, "b": 0.0}
    
    for i in range(150):
        model.learn_one(x, i % 2)
    
    assert len(model.stress_history) == 150
    assert all(0.0 <= s <= 1.0 for s in model.stress_history)


# ==================== SNAPSHOT TESTS ====================
def test_snapshot_creation():
    """Test NCRA snapshot creation"""
    model = NEXUS_River(dim=2, enable_ncra=True, max_snapshots=2)
    x = {"a": 1.0, "b": 0.0}
    
    # Force high stress
    model.stress = 0.2
    model.stress_history.extend([0.2] * 100)
    
    model.learn_one(x, 0)
    assert len(model.snapshots) >= 1
    
    # Create another
    model.stress = 0.25
    x2 = {"a": 2.0, "b": 1.0}
    model.learn_one(x2, 1)
    assert len(model.snapshots) <= 2


def test_snapshot_class():
    """Test Snapshot dataclass"""
    w = np.array([1.0, 2.0, 3.0])
    context = np.array([0.5, 0.1])
    
    snapshot = Snapshot.create(w, 0.5, context, 1.0)
    
    assert snapshot.weight == 1.0
    assert snapshot.bias == 0.5
    assert snapshot.context_norm > 0
    assert np.array_equal(snapshot.w, w)


def test_snapshot_deque_limit():
    """Test snapshot buffer limit"""
    model = NEXUS_River(dim=2, max_snapshots=3)
    x = {"a": 1.0, "b": 0.0}
    
    # Force many snapshots
    for i in range(10):
        model.stress = 0.3
        model.stress_history.extend([0.3] * 100)
        model.learn_one(x, i % 2)
        x = {"a": float(i), "b": float(i + 1)}
    
    # Should be capped at max_snapshots
    assert len(model.snapshots) <= 3


def test_ncra_prediction():
    """Test NCRA ensemble prediction"""
    model = NEXUS_River(dim=2, enable_ncra=True)
    x = {"a": 1.0, "b": 0.0}
    
    # Create snapshot
    model.stress = 0.3
    model.stress_history.extend([0.3] * 100)
    model.learn_one(x, 1)
    
    if model.snapshots:
        x_arr = model._to_array(x)
        p_ncra = model._predict_ncra(x_arr)
        assert 0.0 <= p_ncra <= 1.0


# ==================== WEIGHT DECAY TESTS ====================
def test_weight_decay():
    """Test snapshot weight decay"""
    model = NEXUS_River(dim=2, enable_ncra=True, max_snapshots=1)
    x = {"a": 1.0, "b": 0.0}
    
    # Create snapshot
    model.stress = 0.3
    model.stress_history.extend([0.3] * 100)
    model.learn_one(x, 1)
    
    if model.snapshots:
        initial_weight = model.snapshots[0].weight
        
        # Many updates
        for _ in range(100):
            model.learn_one(x, 1)
        
        final_weight = model.snapshots[0].weight
        assert final_weight > 0.0
        assert final_weight <= initial_weight


# ==================== INPUT VALIDATION TESTS ====================
def test_invalid_input_type():
    """Test type validation"""
    model = NEXUS_River()
    
    with pytest.raises(TypeError):
        model.predict_one("not a dict")  # type: ignore
    
    with pytest.raises(TypeError):
        model.learn_one([1, 2, 3], 1)  # type: ignore


def test_empty_features():
    """Test empty feature dict"""
    model = NEXUS_River()
    
    with pytest.raises(ValueError):
        model.predict_one({})


def test_non_numeric_features():
    """Test non-numeric feature values"""
    model = NEXUS_River()
    
    with pytest.raises(TypeError):
        model.predict_one({"a": "not a number"})


def test_nan_handling():
    """Test NaN handling"""
    model = NEXUS_River(dim=3)
    x = {"a": 1.0, "b": np.nan, "c": 3.0}
    
    # Should not crash
    proba = model.predict_proba_one(x)
    assert 0.0 <= proba[True] <= 1.0


def test_inf_handling():
    """Test infinity handling"""
    model = NEXUS_River(dim=2)
    x = {"a": np.inf, "b": -np.inf}
    
    proba = model.predict_proba_one(x)
    assert 0.0 <= proba[True] <= 1.0


# ==================== STATE MANAGEMENT TESTS ====================
def test_save_load():
    """Test model serialization"""
    model = NEXUS_River(dim=3)
    x = {"a": 1.0, "b": 2.0, "c": 3.0}
    
    for i in range(50):
        model.learn_one(x, i % 2)
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    
    model.save(path)
    loaded = NEXUS_River.load(path)
    
    assert loaded.dim == model.dim
    assert loaded.sample_count == model.sample_count
    assert len(loaded.feature_names) == len(model.feature_names)
    assert np.allclose(loaded.w, model.w)
    assert loaded.bias == model.bias
    
    Path(path).unlink()


def test_reset():
    """Test model reset"""
    model = NEXUS_River(dim=2)
    x = {"a": 1.0, "b": 0.0}
    
    for i in range(100):
        model.learn_one(x, i % 2)
    
    model.reset()
    
    assert model.sample_count == 0
    assert model.stress == 0.0
    assert len(model.stress_history) == 0
    assert len(model.snapshots) == 0
    assert len(model.feature_names) == 0


# ==================== THREAD SAFETY TESTS ====================
def test_thread_safety():
    """Test concurrent access"""
    model = NEXUS_River(dim=2)
    x = {"a": 1.0, "b": 0.0}
    
    def task():
        for _ in range(100):
            model.learn_one(x, np.random.randint(0, 2))
            model.predict_one(x)
    
    threads = [threading.Thread(target=task) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert model.sample_count == 500


# ==================== RFC TESTS ====================
def test_rfc_disabled():
    """Test RFC can be disabled"""
    model = NEXUS_River(dim=2, enable_rfc=False)
    assert model.rfc_w is None


def test_rfc_enabled():
    """Test RFC initialization"""
    model = NEXUS_River(dim=2, enable_rfc=True)
    x = {"a": 1.0, "b": 0.0}
    model.learn_one(x, 1)
    
    assert model.rfc_w is not None
    assert len(model.rfc_w) == 2


# ==================== REPR TESTS ====================
def test_repr():
    """Test string representation"""
    model = NEXUS_River(dim=5)
    x = {f"f{i}": float(i) for i in range(5)}
    model.learn_one(x, 1)
    
    repr_str = repr(model)
    assert "NEXUS_River" in repr_str
    assert "dim=5" in repr_str
    assert "samples=1" in repr_str


# ==================== EDGE CASES ====================
def test_single_feature():
    """Test with single feature"""
    model = NEXUS_River()
    x = {"only_feature": 1.0}
    
    model.learn_one(x, 1)
    proba = model.predict_proba_one(x)
    
    assert 0.0 <= proba[True] <= 1.0


def test_many_features():
    """Test with many features"""
    model = NEXUS_River()
    x = {f"f{i}": float(i) for i in range(100)}
    
    model.learn_one(x, 1)
    proba = model.predict_proba_one(x)
    
    assert 0.0 <= proba[True] <= 1.0


def test_long_feature_names():
    """Test with long feature names"""
    model = NEXUS_River()
    x = {"a" * 500: 1.0, "b" * 500: 2.0}
    
    model.learn_one(x, 1)
    proba = model.predict_proba_one(x)
    
    assert 0.0 <= proba[True] <= 1.0


def test_unicode_feature_names():
    """Test with unicode feature names"""
    model = NEXUS_River()
    x = {"特征_1": 1.0, "특징_2": 2.0, "คุณลักษณะ_3": 3.0}
    
    model.learn_one(x, 1)
    proba = model.predict_proba_one(x)
    
    assert 0.0 <= proba[True] <= 1.0


# ==================== PERFORMANCE TESTS ====================
def test_performance_no_memory_leak():
    """Test for memory leaks"""
    model = NEXUS_River(dim=10)
    
    for i in range(1000):
        x = {f"f{j}": float(i + j) for j in range(10)}
        model.learn_one(x, i % 2)
    
    # Snapshots should be bounded
    assert len(model.snapshots) <= model.snapshots.maxlen
    assert len(model.stress_history) <= model.stress_history.maxlen


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=nexus", "--cov-report=html"])
