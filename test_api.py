import traceback
import json
import numpy as np
import pandas as pd

import ml_studio
from ml_studio.api import Project

try:
    print("1. Create project")
    p = Project.create("fraud_test")
    p.task = "classification"

    print("\n2. Load dataset")
    # Generate a realistic 5M row dataset
    print("Generating 5M row dataset...")
    df = pd.DataFrame({
        'feature1': np.random.randn(5_000_000),
        'feature2': np.random.rand(5_000_000),
        'categorical': np.random.choice(['A', 'B', 'C'], 5_000_000),
        'MedHouseVal': np.random.choice([0, 1], 5_000_000)
    })
    df.to_csv('synthetic_5M.csv', index=False)
    p.load_data("synthetic_5M.csv")

    print("\n3. Profile it")
    profile = p.profile()
    
    print("Profile JSON:")
    print(json.dumps(profile, indent=2)[:500] + "...\n")
    
    print(f"\nQuality Score: {profile.get('quality_score', 'MISSING')}")
    print(f"Leakage Warnings: {profile.get('leakage_warnings', 'MISSING')}")

    print("\n4. Auto-sample check")
    if hasattr(p, 'sample_info'):
        print(p.sample_info())
    else:
        print("sample_info: MISSING")

    print("\n5. Set target")
    p.set_target("MedHouseVal")

    print("\n6. Access prepared data")
    if hasattr(p, 'get_xy'):
        X, y = p.get_xy()
        print(X.shape, y.shape)
    else:
        print("get_xy: MISSING")

    print("\n7. Save + reload")
    if hasattr(p, 'save'):
        p.save()
        p2 = Project.open("fraud_test")
        assert p2.target == "MedHouseVal"
        print("Persistence works!")
    else:
        print("save/load: MISSING")

except Exception as e:
    traceback.print_exc()
