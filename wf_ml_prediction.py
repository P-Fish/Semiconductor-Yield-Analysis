import warnings
import os
from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score


def evaluate_models(X_test, y_test, base_path="models"):
    """
    Evaluate models with warnings suppressed and parallel processing disabled.
    Creates a human-readable summary file.
    """
    # Disable parallel processing globally for scikit-learn
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    os.environ["JOBLIB_PARALLEL"] = "0"

    model_dir = Path(base_path)
    evaluation_dir = model_dir.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    results = []
    pos_label = -1
    summary_lines = []

    # Add header to summary
    summary_lines.append("MODEL EVALUATION SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Test Set Size: {len(y_test)}")
    summary_lines.append("=" * 50 + "\n")

    # Suppress all warnings during evaluation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for model_path in model_dir.glob("*.pkl"):
            if model_path.stem == 'scaler':
                continue

            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                # Ensure model's predict method doesn't use parallel processing
                if hasattr(model, 'n_jobs'):
                    model.n_jobs = None

                y_pred = model.predict(X_test)

                # Calculate metrics
                metrics = {
                    'model_name': model_path.stem,
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
                    'recall': recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
                    'f1': f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
                }

                results.append(metrics)

                # Calculate confusion matrix information
                passed_correct = sum((y_test == -1) & (y_pred == -1))
                passed_incorrect = sum((y_test == -1) & (y_pred == 1))
                failed_correct = sum((y_test == 1) & (y_pred == 1))
                failed_incorrect = sum((y_test == 1) & (y_pred == -1))

                # Add model-specific results to summary
                summary_lines.append(f"MODEL: {model_path.stem.upper()}")
                summary_lines.append("-" * 50)
                summary_lines.append("Performance Metrics:")
                summary_lines.append(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
                summary_lines.append(f"  Accuracy:         {metrics['accuracy']:.3f}")
                summary_lines.append(f"  Precision:        {metrics['precision']:.3f}")
                summary_lines.append(f"  Recall:           {metrics['recall']:.3f}")
                summary_lines.append(f"  F1 Score:         {metrics['f1']:.3f}")

                summary_lines.append("\nConfusion Matrix Details:")
                summary_lines.append("  Passes (-1):")
                summary_lines.append(f"    Correctly predicted:   {passed_correct}")
                summary_lines.append(f"    Incorrectly predicted: {passed_incorrect}")
                summary_lines.append("  Failures (1):")
                summary_lines.append(f"    Correctly predicted:   {failed_correct}")
                summary_lines.append(f"    Incorrectly predicted: {failed_incorrect}")
                summary_lines.append("\n" + "=" * 50 + "\n")

                # Print to console
                print(f"\nModel: {model_path.stem}")
                print(f"Correctly predicted passes (-1): {passed_correct}")
                print(f"Incorrectly predicted passes: {passed_incorrect}")
                print(f"Correctly predicted failures (1): {failed_correct}")
                print(f"Incorrectly predicted failures: {failed_incorrect}")

            except Exception as e:
                print(f"Error evaluating {model_path.stem}: {str(e)}")
                continue

    # Save detailed summary
    try:
        with open(evaluation_dir / "summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
    except Exception as e:
        print(f"Warning: Could not save evaluation results: {str(e)}")

    results_df = pd.DataFrame(results)

    return results_df