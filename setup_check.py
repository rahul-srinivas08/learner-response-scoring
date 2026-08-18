"""Environment setup + sanity check for Part 1/2/3.

Run before opening any notebook: `python3 setup_check.py`
Checks (1) every required package is installed and importable, (2) the two
project modules (`feature_engineering.py`, `evaluation.py`) work correctly
on a small smoke test, not just that they import.

The OMP_NUM_THREADS / KMP_DUPLICATE_LIB_OK env vars below fix a real crash
found while building Part 3: running XGBoost's .fit() in the same process
after a PyTorch/LoRA training loop segfaults on this platform (a known
torch/xgboost OpenMP conflict). They must be set before torch or xgboost
are imported anywhere in a process -- that's why this file sets them at
the very top, and why Part 3's notebook does the same in its first cell.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import importlib
import sys

REQUIRED = {
    "pandas": "Part 1/2/3", "numpy": "Part 1/2/3", "matplotlib": "Part 1/2/3",
    "seaborn": "Part 1", "sklearn": "Part 1/2/3 (pip name: scikit-learn)",
    "scipy": "Part 1/2/3", "lightgbm": "Part 1", "shap": "Part 1",
    "sentence_transformers": "Part 1/2/3 (pip name: sentence-transformers)",
    "xgboost": "Part 2/3", "torch": "Part 3", "transformers": "Part 3",
    "peft": "Part 3 (LoRA)",
}

REQUIRED_FILES = ["dataset.csv", "feature_engineering.py", "evaluation.py"]


def check_packages() -> list[str]:
    print(f"Python: {sys.executable}  ({sys.version.split()[0]})\n")
    missing = []
    for mod, used_by in REQUIRED.items():
        try:
            m = importlib.import_module(mod)
            print(f"  OK    {mod:24s} {getattr(m, '__version__', '?'):12s} ({used_by})")
        except ImportError:
            print(f"  MISSING {mod:22s} needed for {used_by}")
            missing.append(mod)
    return missing


def check_files() -> list[str]:
    print("\nRequired files in current directory:")
    missing = []
    for f in REQUIRED_FILES:
        ok = os.path.exists(f)
        print(f"  {'OK' if ok else 'MISSING':7s} {f}")
        if not ok:
            missing.append(f)
    return missing


def check_eval_smoke_test() -> bool:
    """Not just 'does it import' -- does evaluation.py actually compute what
    Part 1/2/3 assume it computes? Catches a broken/mismatched environment
    before it silently produces wrong numbers 20 minutes into a notebook run."""
    print("\nSmoke-testing evaluation.py and feature_engineering.py:")
    ok = True
    try:
        import numpy as np
        import pandas as pd
        from evaluation import evaluate, HUMAN_QWK_CEIL, HUMAN_MAE_CEIL, ORACLE_MAE_CEIL, action_of

        y = np.array([0, 1, 2, 3, 4, 2, 2, 1])
        perfect = evaluate(y, y)
        majority = evaluate(y, np.full(len(y), 2))
        checks = [
            ("perfect predictions -> QWK == 1.0", perfect["qwk"] == 1.0),
            ("perfect predictions -> MAE == 0.0", perfect["mae"] == 0.0),
            ("constant predictions -> QWK == 0.0", majority["qwk"] == 0.0),
            ("action_of(0)==action_of(1) (both re-prompt)", action_of(0) == action_of(1)),
            ("action_of(2) != action_of(3)", action_of(2) != action_of(3)),
            ("HUMAN_QWK_CEIL is Part 1's computed 0.795", HUMAN_QWK_CEIL == 0.795),
            ("HUMAN_MAE_CEIL is Part 1's computed 0.484", HUMAN_MAE_CEIL == 0.484),
            ("ORACLE_MAE_CEIL is Part 1's computed 0.743", ORACLE_MAE_CEIL == 0.743),
        ]
        for name, passed in checks:
            print(f"  {'OK' if passed else 'FAIL':6s} {name}")
            ok = ok and passed

        from feature_engineering import build_features, CEFR_ORDINAL
        row = pd.DataFrame({
            "prompt": ["Talk about your favorite hobby."], "target_language": ["en"],
            "cefr_level": ["B1"], "asr_transcript": ["i like reading books"],
            "asr_word_confidences": ["[0.9, 0.8, 0.85, 0.9]"],
        })
        feats = build_features(row)
        checks2 = [
            ("build_features returns 28 columns", feats.shape[1] == 28),
            ("build_features runs on a single row (serving-safe)", feats.shape[0] == 1),
            ("C2 encodes to ordinal 5 even though unseen in dataset.csv", CEFR_ORDINAL["C2"] == 5),
        ]
        for name, passed in checks2:
            print(f"  {'OK' if passed else 'FAIL':6s} {name}")
            ok = ok and passed
    except Exception as e:
        print(f"  FAIL  smoke test raised an exception: {type(e).__name__}: {e}")
        ok = False
    return ok


if __name__ == "__main__":
    missing_pkgs = check_packages()
    missing_files = check_files()
    eval_ok = check_eval_smoke_test() if not missing_pkgs and not missing_files else False

    print("\n" + "=" * 60)
    if missing_pkgs:
        print(f"FAIL: missing packages: {', '.join(missing_pkgs)}")
        print("      fix: pip install -r requirements.txt")
    if missing_files:
        print(f"FAIL: missing files: {', '.join(missing_files)}")
        print("      run this script from the project root (same folder as dataset.csv)")
    if not missing_pkgs and not missing_files:
        print("PASS: environment ready" if eval_ok else "FAIL: evaluation.py/feature_engineering.py smoke test did not pass")
    sys.exit(0 if (not missing_pkgs and not missing_files and eval_ok) else 1)
