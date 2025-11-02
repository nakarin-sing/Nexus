#!/usr/bin/env python3
"""
NEXUS Quickstart Example
Demonstrates basic usage with River datasets
"""

from river import datasets, metrics, preprocessing
from nexus import NEXUS_River
import matplotlib.pyplot as plt


def basic_example():
    """Basic classification example"""
    print("=" * 60)
    print("NEXUS v4.0.1 - Basic Example")
    print("=" * 60)
    
    # Create model
    model = NEXUS_River(enable_ncra=True, enable_rfc=True)
    metric = metrics.ROCAUC()
    
    # Train on Phishing dataset
    print("\nTraining on Phishing dataset...")
    for i, (x, y) in enumerate(datasets.Phishing().take(1000)):
        y_pred = model.predict_proba_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred[True])
        
        if (i + 1) % 200 == 0:
            print(f"  Sample {i+1:4d} | AUC: {metric.get():.4f} | Stress: {model.stress:.4f}")
    
    print(f"\nFinal AUC: {metric.get():.4f}")
    print(f"Snapshots: {len(model.snapshots)}")
    print(f"Features: {len(model.feature_names)}")


def pipeline_example():
    """Example with preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("NEXUS with Preprocessing Pipeline")
    print("=" * 60)
    
    # Create pipeline
    model = preprocessing.StandardScaler() | NEXUS_River()
    metric = metrics.Accuracy()
    
    print("\nTraining on Electricity dataset...")
    for i, (x, y) in enumerate(datasets.Elec2().take(1000)):
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred)
        
        if (i + 1) % 200 == 0:
            print(f"  Sample {i+1:4d} | Accuracy: {metric.get():.4f}")
    
    print(f"\nFinal Accuracy: {metric.get():.4f}")


def stress_monitoring_example():
    """Example showing stress monitoring"""
    print("\n" + "=" * 60)
    print("Stress Monitoring Example")
    print("=" * 60)
    
    model = NEXUS_River()
    
    print("\nTraining and monitoring stress...")
    for i, (x, y) in enumerate(datasets.Phishing().take(500)):
        model.learn_one(x, y)
        
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1:3d} | Stress: {model.stress:.4f} | Snapshots: {len(model.snapshots)}")
    
    # Plot stress history
    if len(model.stress_history) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(list(model.stress_history))
        plt.title("Model Stress Over Time")
        plt.xlabel("Sample")
        plt.ylabel("Stress")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("stress_history.png", dpi=150)
        print("\n✓ Stress plot saved to stress_history.png")


def save_load_example():
    """Example of model persistence"""
    print("\n" + "=" * 60)
    print("Save/Load Example")
    print("=" * 60)
    
    # Train model
    model = NEXUS_River()
    print("\nTraining model...")
    for i, (x, y) in enumerate(datasets.Phishing().take(500)):
        model.learn_one(x, y)
    
    print(f"Original model: {model}")
    
    # Save
    model.save("model.pkl")
    print("✓ Model saved to model.pkl")
    
    # Load
    loaded_model = NEXUS_River.load("model.pkl")
    print(f"✓ Model loaded: {loaded_model}")
    
    # Verify
    x_test = {f"f{i}": float(i) for i in range(10)}
    p1 = model.predict_proba_one(x_test)
    p2 = loaded_model.predict_proba_one(x_test)
    
    print(f"\nOriginal prediction: {p1[True]:.4f}")
    print(f"Loaded prediction:   {p2[True]:.4f}")
    print(f"Match: {abs(p1[True] - p2[True]) < 1e-6}")


def comparison_example():
    """Compare NEXUS with baseline"""
    print("\n" + "=" * 60)
    print("Comparison with Baseline")
    print("=" * 60)
    
    from river import linear_model
    
    # NEXUS
    model_nexus = NEXUS_River()
    metric_nexus = metrics.ROCAUC()
    
    # Baseline
    model_baseline = linear_model.LogisticRegression()
    metric_baseline = metrics.ROCAUC()
    
    print("\nTraining both models on CreditCard dataset...")
    for i, (x, y) in enumerate(datasets.CreditCard().take(1000)):
        # NEXUS
        y_pred = model_nexus.predict_proba_one(x)
        model_nexus.learn_one(x, y)
        metric_nexus.update(y, y_pred[True])
        
        # Baseline
        y_pred = model_baseline.predict_proba_one(x)
        model_baseline.learn_one(x, y)
        metric_baseline.update(y, y_pred[True])
        
        if (i + 1) % 200 == 0:
            print(f"  Sample {i+1:4d} | NEXUS: {metric_nexus.get():.4f} | Baseline: {metric_baseline.get():.4f}")
    
    print(f"\nFinal NEXUS AUC:    {metric_nexus.get():.4f}")
    print(f"Final Baseline AUC: {metric_baseline.get():.4f}")
    print(f"Improvement: {(metric_nexus.get() - metric_baseline.get())*100:+.2f}%")


def main():
    """Run all examples"""
    examples = [
        ("Basic Usage", basic_example),
        ("Pipeline", pipeline_example),
        ("Stress Monitoring", stress_monitoring_example),
        ("Save/Load", save_load_example),
        ("Comparison", comparison_example),
    ]
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "NEXUS QUICKSTART EXAMPLES" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
