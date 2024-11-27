import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def conduct_feature_experiments(model, X_test, scaler=None, output_dir="evaluation/experiment_results"):
    """
    Conduct experiments varying feature_516 and feature_244 values to analyze their impact on predictions.

    Parameters:
    model: trained sklearn model
    X_test: pandas DataFrame with test data
    scaler: optional StandardScaler used in training
    output_dir: directory to save visualization results

    Returns:
    dict containing experiment results and analysis
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create synthetic data points for experimentation
    def create_synthetic_point(base_point, feature_516_val, feature_244_val):
        new_point = base_point.copy()
        new_point['feature_516'] = feature_516_val
        new_point['feature_244'] = feature_244_val
        return new_point

    # Get feature ranges from test data
    feature_516_range = np.linspace(X_test['feature_516'].min(), X_test['feature_516'].max(), 50)
    feature_244_range = np.linspace(X_test['feature_244'].min(), X_test['feature_244'].max(), 50)

    # Base point for experimentation (using median values)
    base_point = pd.DataFrame([X_test.median()], columns=X_test.columns)

    # Experiment 1: Varying feature_516
    feature_516_results = []
    for val in feature_516_range:
        test_point = create_synthetic_point(base_point, val, base_point['feature_244'].iloc[0])
        if scaler:
            test_point_scaled = pd.DataFrame(scaler.transform(test_point), columns=test_point.columns)
            pred = model.predict_proba(test_point_scaled)[0]
        else:
            pred = model.predict_proba(test_point)[0]
        feature_516_results.append({'value': val, 'pass_prob': pred[0]})

    # Experiment 2: Varying feature_244
    feature_244_results = []
    for val in feature_244_range:
        test_point = create_synthetic_point(base_point, base_point['feature_516'].iloc[0], val)
        if scaler:
            test_point_scaled = pd.DataFrame(scaler.transform(test_point), columns=test_point.columns)
            pred = model.predict_proba(test_point_scaled)[0]
        else:
            pred = model.predict_proba(test_point)[0]
        feature_244_results.append({'value': val, 'pass_prob': pred[0]})

    # Experiment 3: Varying both features
    heatmap_data = np.zeros((len(feature_516_range), len(feature_244_range)))
    for i, val_516 in enumerate(feature_516_range):
        for j, val_244 in enumerate(feature_244_range):
            test_point = create_synthetic_point(base_point, val_516, val_244)
            if scaler:
                test_point_scaled = pd.DataFrame(scaler.transform(test_point), columns=test_point.columns)
                pred = model.predict_proba(test_point_scaled)[0]
            else:
                pred = model.predict_proba(test_point)[0]
            heatmap_data[i, j] = pred[0]

    # Create visualizations
    # Individual feature plots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot([x['value'] for x in feature_516_results],
             [x['pass_prob'] for x in feature_516_results])
    plt.title('Feature 516 Impact on Pass Probability')
    plt.xlabel('Feature 516 Value')
    plt.ylabel('Pass Probability')

    plt.subplot(1, 2, 2)
    plt.plot([x['value'] for x in feature_244_results],
             [x['pass_prob'] for x in feature_244_results])
    plt.title('Feature 244 Impact on Pass Probability')
    plt.xlabel('Feature 244 Value')
    plt.ylabel('Pass Probability')

    plt.tight_layout()
    plt.savefig(output_path / 'individual_feature_impacts.png')
    plt.close()

    # Heatmap for combined effects
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data,
                xticklabels=np.round(feature_244_range[::5], 2),
                yticklabels=np.round(feature_516_range[::5], 2),
                cmap='coolwarm')
    plt.title('Combined Feature Impact on Pass Probability')
    plt.xlabel('Feature 244 Value')
    plt.ylabel('Feature 516 Value')
    plt.tight_layout()
    plt.savefig(output_path / 'feature_interaction_heatmap.png')
    plt.close()

    # Calculate results
    results = {
        'feature_516': {
            'min_effect': min(x['pass_prob'] for x in feature_516_results),
            'max_effect': max(x['pass_prob'] for x in feature_516_results),
            'range': max(x['pass_prob'] for x in feature_516_results) -
                     min(x['pass_prob'] for x in feature_516_results)
        },
        'feature_244': {
            'min_effect': min(x['pass_prob'] for x in feature_244_results),
            'max_effect': max(x['pass_prob'] for x in feature_244_results),
            'range': max(x['pass_prob'] for x in feature_244_results) -
                     min(x['pass_prob'] for x in feature_244_results)
        },
        'combined': {
            'min_effect': np.min(heatmap_data),
            'max_effect': np.max(heatmap_data),
            'range': np.max(heatmap_data) - np.min(heatmap_data)
        }
    }

    # Create summary lines for text file
    summary_lines = [
        "FEATURE EXPERIMENTATION SUMMARY",
        "=" * 80,
        f"Experiment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80 + "\n",

        "EXPERIMENT 1: VARYING FEATURE 516",
        "-" * 80,
        f"Minimum Pass Probability: {results['feature_516']['min_effect']:.3f}",
        f"Maximum Pass Probability: {results['feature_516']['max_effect']:.3f}",
        f"Effect Range: {results['feature_516']['range']:.3f}",
        "Analysis:",
        f"  -> Feature 516 can swing predictions by {results['feature_516']['range'] * 100:.1f}%",
        f"  -> Strongest positive effect at value {feature_516_range[np.argmax([x['pass_prob'] for x in feature_516_results])]:.2f}",
        "\n" + "=" * 80 + "\n",

        "EXPERIMENT 2: VARYING FEATURE 244",
        "-" * 80,
        f"Minimum Pass Probability: {results['feature_244']['min_effect']:.3f}",
        f"Maximum Pass Probability: {results['feature_244']['max_effect']:.3f}",
        f"Effect Range: {results['feature_244']['range']:.3f}",
        "Analysis:",
        f"  -> Feature 244 can swing predictions by {results['feature_244']['range'] * 100:.1f}%",
        f"  -> Strongest positive effect at value {feature_244_range[np.argmax([x['pass_prob'] for x in feature_244_results])]:.2f}",
        "\n" + "=" * 80 + "\n",

        "EXPERIMENT 3: VARYING BOTH FEATURES",
        "-" * 80,
        f"Minimum Combined Pass Probability: {results['combined']['min_effect']:.3f}",
        f"Maximum Combined Pass Probability: {results['combined']['max_effect']:.3f}",
        f"Combined Effect Range: {results['combined']['range']:.3f}",
        "Analysis:",
        f"  -> Combined features can swing predictions by {results['combined']['range'] * 100:.1f}%",
        f"  -> Interaction strength: {(results['combined']['range'] - max(results['feature_516']['range'], results['feature_244']['range'])) * 100:.1f}% additional effect",
        "\n" + "=" * 80 + "\n",
    ]

    # Save summary
    with open(output_path / "experiment_summary.txt", 'w') as f:
        f.write('\n'.join(summary_lines))

    return results