import warnings
from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score


def evaluate_models(X_test, y_test, base_path="models"):
    """
    Evaluate models with warnings suppressed.
    """
    model_dir = Path(base_path)
    evaluation_dir = model_dir.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    results = []
    pos_label = -1

    # Suppress all warnings during evaluation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for model_path in model_dir.glob("*.pkl"):
            if model_path.stem == 'scaler':
                continue

            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                y_pred = model.predict(X_test)

                metrics = {
                    'model_name': model_path.stem,
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
                    'recall': recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
                    'f1': f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
                }

                results.append(metrics)

                # Print confusion matrix information
                passed_correct = sum((y_test == -1) & (y_pred == -1))
                passed_incorrect = sum((y_test == -1) & (y_pred == 1))
                failed_correct = sum((y_test == 1) & (y_pred == 1))
                failed_incorrect = sum((y_test == 1) & (y_pred == -1))

                print(f"\nModel: {model_path.stem}")
                print(f"Correctly predicted passes (-1): {passed_correct}")
                print(f"Incorrectly predicted passes: {passed_incorrect}")
                print(f"Correctly predicted failures (1): {failed_correct}")
                print(f"Incorrectly predicted failures: {failed_incorrect}")

            except Exception as e:
                print(f"Error evaluating {model_path.stem}: {str(e)}")
                continue

    results_df = pd.DataFrame(results)

    try:
        results_df.to_csv(evaluation_dir / "summary.txt", index=False)
    except Exception as e:
        print(f"Warning: Could not save evaluation results: {str(e)}")

    return results_df
