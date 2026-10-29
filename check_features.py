# check_features.py
import pickle
from pathlib import Path

MODELS_DIR = Path("models")

def check_model(name):
    path = MODELS_DIR / name
    if not path.exists():
        print(f"❌ {name} — НЕ СЪЩЕСТВУВА")
        return
    
    print(f"\n{'=' * 70}")
    print(f"📦 {name}")
    print(f"{'=' * 70}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Metadata
    print(f"\nMetadata:")
    print(f"  model_name: {data.get('model_name', 'N/A')}")
    print(f"  fit_mape:   {data.get('fit_mape', 'N/A')}")
    print(f"  baseline_mape: {data.get('baseline_mape', 'N/A')}")
    
    # Features
    feature_cols = data.get("feature_cols", [])
    print(f"\nFeatures ({len(feature_cols)}):")
    for i, f in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {f}")


# Провери всички модели
check_model("production_model.pkl")
check_model("boosting_stable.pkl")

print(f"\n{'=' * 70}")
print("✅ Проверката завърши!")
print(f"{'=' * 70}")