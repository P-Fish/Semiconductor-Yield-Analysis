#### SER494: Machine Learning Evaluation
#### Semiconductor Yield Prediction
#### Peter Fischbach
#### 11/26/24

## Evaluation Metrics
### Metric 1
**Name:** Balanced Accuracy

**Choice Justification:** Our dataset has imbalanced classes (unequal numbers of passes and fails). Balanced accuracy accounts for this imbalance by averaging the recall obtained on each class, providing a more reliable performance measure than standard accuracy when working with imbalanced datasets.

**Interpretation:** A balanced accuracy of 0.875 means the model is correct 87.5% of the time when accounting for class imbalance. This metric helps us understand if the model is performing well on both passes and fails, rather than just doing well on the majority class.

### Metric 2
**Name:** F1 Score

**Choice Justification:** In manufacturing quality control, we need to balance between identifying actual failures (recall) and not misclassifying good products as failures (precision). F1 score provides this balance by taking the harmonic mean of precision and recall.

**Interpretation:** An F1 score of 0.875 indicates good balance between precision and recall. This means the model is effective at both identifying true failures and minimizing false alarms.

## Alternative Models
### Alternative 1
**Construction:** Random Forest Classifier
- 200 trees
- Max depth of 8
- Minimum 10 samples per leaf
- Class weights adjusted for imbalance
- Single-threaded processing

**Evaluation:** 
- Strong balanced accuracy (0.875)
- Good handling of feature relationships
- Less prone to overfitting due to ensemble nature
- Interpretable feature importance rankings

### Alternative 2
**Construction:** Gradient Boosting Classifier
- 300 trees
- Learning rate of 0.05
- Max depth of 4
- 80% subsample rate
- Conservative tree complexity

**Evaluation:**
- Slightly higher precision than Random Forest
- More computationally intensive
- Better handling of subtle patterns
- More sensitive to hyperparameter tuning

### Alternative 3
**Construction:** Logistic Regression
- L2 regularization (C=0.1)
- Class weights for imbalance
- LibLinear solver
- 2000 max iterations

**Evaluation:**
- More interpretable coefficients
- Faster training time
- Less complex model
- Lower overall performance than tree-based models

## Visualization
### Visual 1
**Analysis:** The confusion matrix reveals that false positives (predicting pass when actually fail) are more common than false negatives for most models. This suggests we might need to adjust our decision threshold or class weights to be more conservative in pass predictions.
<br>
<img alt="random_forest_confusion_matrix.png" src="evaluation%2Fvisualizations%2Frandom_forest_confusion_matrix.png" height="280"/>
<img alt="logistic_regression_confusion_matrix.png" src="evaluation%2Fvisualizations%2Flogistic_regression_confusion_matrix.png" height="280"/>
<img alt="knn_confusion_matrix.png" src="evaluation%2Fvisualizations%2Fknn_confusion_matrix.png" height="280"/>
<img alt="gradient_boosting_confusion_matrix.png" src="evaluation%2Fvisualizations%2Fgradient_boosting_confusion_matrix.png" height="280"/>
<br><br>

### Visual 2
**Analysis:** Feature importance plots from the Gradient Boosting model shows that certain features have significantly higher importance. This insight can be used to focus quality control efforts on monitoring and controlling these specific parameters.
<br>
<img alt="gradient_boosting_feature_importance.png" src="evaluation%2Fvisualizations%2Fgradient_boosting_feature_importance.png" height="360"/>
<br><br>

### Visual 3
**Analysis:** Model performance comparison shows that tree-based models have similar overall performance, while logistic regression makes different types of mistakes. This suggests that incorporating logistic regression and another model could produce better results.
<br>
<img alt="model_comparison.png" src="evaluation%2Fvisualizations%2Fmodel_comparison.png" height="360"/>
<br><br>

## Best Model

**Model:** Gradient Boosting
- Best or at least equivalent Balanced Accuracy
- Better precision than Random Forest, which is crucial for manufacturing quality control to minimize false passes
- Excellent handling of non-linear feature relationships and complex patterns
- Strong performance on imbalanced datasets
- Feature importance rankings provide valuable process insights
- While more computationally intensive during training, prediction speed remains fast
- Higher sensitivity to hyperparameters allows for better fine-tuning to specific manufacturing conditions