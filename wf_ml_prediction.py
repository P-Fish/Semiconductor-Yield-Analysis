import warnings
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, \
    confusion_matrix


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Create and save a confusion matrix visualization."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)

    # Create confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pass (-1)', 'Fail (1)'],
                yticklabels=['Pass (-1)', 'Fail (1)'])

    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / f'{model_name.lower()}_confusion_matrix.png')
    plt.close()


def plot_feature_importance(model, feature_names, model_name, save_path):
    """Create and save feature importance visualization."""
    plt.figure(figsize=(10, 6))

    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])

    if importances is not None:
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        top_n = 10  # Show top 10 features

        # Create bar plot
        plt.bar(range(top_n), importances[indices][:top_n])
        plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=45, ha='right')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(save_path / f'{model_name.lower()}_feature_importance.png')
        plt.close()


def plot_model_comparison(results_df, save_path):
    """Create and save model comparison visualization."""
    plt.figure(figsize=(12, 6))

    metrics = ['balanced_accuracy', 'accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results_df['model_name']))
    width = 0.15

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, results_df[metric], width, label=metric.replace('_', ' ').title())

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * 2, results_df['model_name'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / 'model_comparison.png')
    plt.close()


def evaluate_models(X_test, y_test, base_path="models"):
    """
    Evaluate models with warnings suppressed and create visualizations.
    """
    # Disable parallel processing globally for scikit-learn
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    os.environ["JOBLIB_PARALLEL"] = "0"

    model_dir = Path(base_path)
    evaluation_dir = model_dir.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization directory
    viz_dir = evaluation_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    results = []
    pos_label = -1
    summary_lines = []

    # Add header to summary
    summary_lines.append("MODEL EVALUATION SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Test Set Size: {len(y_test)}")
    summary_lines.append("=" * 80 + "\n")

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

                # Create visualizations
                plot_confusion_matrix(y_test, y_pred, model_path.stem, viz_dir)
                plot_feature_importance(model, X_test.columns, model_path.stem, viz_dir)

                # Add results to summary
                summary_lines.extend([
                    f"MODEL: {model_path.stem.upper()}",
                    "-" * 80,
                    "Performance Metrics:",
                    f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}",
                    f"    -> Model is correct {metrics['balanced_accuracy'] * 100:.1f}% of the time when accounting for class imbalance",
                    f"\n  Accuracy: {metrics['accuracy']:.3f}",
                    f"    -> {metrics['accuracy'] * 100:.1f}% of all predictions are correct",
                    f"\n  Precision: {metrics['precision']:.3f}",
                    f"    -> When model predicts 'pass', it's right {metrics['precision'] * 100:.1f}% of the time",
                    f"\n  Recall: {metrics['recall']:.3f}",
                    f"    -> Model correctly identifies {metrics['recall'] * 100:.1f}% of actual passes",
                    f"\n  F1 Score: {metrics['f1']:.3f}",
                    f"    -> Overall balance of precision and recall",
                    "\n" + "=" * 80 + "\n"
                ])

            except Exception as e:
                print(f"Error evaluating {model_path.stem}: {str(e)}")
                continue

    # Create model comparison plot
    results_df = pd.DataFrame(results)
    plot_model_comparison(results_df, viz_dir)

    # Save summary
    try:
        with open(evaluation_dir / "summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
    except Exception as e:
        print(f"Warning: Could not save evaluation results: {str(e)}")

    return results_df