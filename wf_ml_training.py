import warnings
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def get_feature_importance(model, feature_names):
    """
    Get feature importance from the model if available.
    """
    try:
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models like LogisticRegression
            return dict(zip(feature_names, np.abs(model.coef_[0])))
        else:
            # For models without feature importance (like KNN)
            return None
    except:
        return None


def train_classification_models(X_train, y_train, base_path="models"):
    """
    Train multiple classification models with parallel processing completely disabled.
    """
    n_samples = len(y_train)
    n_passed = sum(y_train == -1)
    n_failed = sum(y_train == 1)

    weight_for_passed = (1 / n_passed) * (n_samples / 2)
    weight_for_failed = (1 / n_failed) * (n_samples / 2) * 1.2
    class_weight = {-1: weight_for_passed, 1: weight_for_failed}

    print(f"Class distribution - Passed (-1): {n_passed}, Failed (1): {n_failed}")
    print(f"Class weights - Passed: {weight_for_passed:.2f}, Failed: {weight_for_failed:.2f}")

    model_dir = Path(base_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Models with ALL parallel processing disabled
    models = {
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.8,
            # GradientBoosting doesn't use parallel processing by default
        ),
        'logistic_regression': LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            C=0.1,
            n_jobs=None,  # Changed from 1 to None to disable parallel processing
            solver='liblinear'  # Using liblinear solver which doesn't use joblib
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight=class_weight,
            n_jobs=None,  # Changed from 1 to None to disable parallel processing
            bootstrap=True
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=min(int(np.sqrt(len(X_train))), 20),
            weights='distance',
            metric='manhattan',
            n_jobs=None,  # Changed from 1 to None to disable parallel processing
            algorithm='auto'
        )
    }

    model_info = {}

    for model_name, model in models.items():
        try:
            print(f"\nTraining {model_name}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            model_path = model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Store model info
            feature_importance = get_feature_importance(model, X_train.columns)
            model_info[model_name] = {
                'path': str(model_path),
                'type': type(model).__name__,
                'parameters': {
                    k: str(v) if isinstance(v, (np.ndarray, list)) else v
                    for k, v in model.get_params().items()
                },
                'feature_importance': feature_importance
            }
            print(f"Completed training {model_name}")

        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue

    try:
        info_path = model_dir / "model_info.json"
        pd.DataFrame(model_info).to_json(info_path)
    except Exception as e:
        print(f"Warning: Could not save model info: {str(e)}")

    return model_info
