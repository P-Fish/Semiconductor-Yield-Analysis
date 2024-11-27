#### SER494: Experimentation
#### Semiconductor Yield Prediction
#### Peter Fischbach
#### 11/26/24

## Explainable Records
### Record 1
**Raw Data:** 'data_original/secom.data' (LINE 1356) Sample with high feature_516 value (consistently top feature across gradient boosting and random forest)

**Prediction Explanation:** The model confidently predicts "Pass" for this case, which aligns with our extremely high recall rates (0.997 for gradient boosting, 1.000 for random forest). The consistency of this feature's importance across multiple models (top in both gradient boosting and random forest) suggests it captures fundamental characteristics that reliably indicate passing cases.

### Record 2
**Raw Data:** 'data_original/secom.data' (LINE 30) Sample with moderate feature_244 value but low feature_516 value

**Prediction Explanation:** This combination yields a "Fail" prediction, supported by the balanced accuracy scores (0.65 for gradient boosting). The model's higher uncertainty on fail cases (visible in confusion matrices) suggests this prediction relies on more subtle feature interactions, as shown by the distributed importance weights in the random forest model.

## Interesting Features
### Feature A
**Feature:** feature_516

**Justification:** This feature shows remarkable consistency across models (top feature in both gradient boosting and random forest), with importance scores significantly higher than other features. Its prominence in multiple model architectures (0.15 in gradient boosting, 0.055 in random forest) suggests it captures a fundamental aspect of the classification boundary.

### Feature B
**Feature:** feature_244

**Justification:** Appears as a strong secondary feature across models, with particularly stable importance scores (second-highest in random forest). Its consistent high ranking suggests it provides complementary information to feature_516, helping achieve the high precision scores (0.948 for gradient boosting, 0.945 for random forest).

## Experiments 
<img alt="individual_feature_impacts.png" src="evaluation%2Fexperiment_results%2Findividual_feature_impacts.png" height="320"/>

### Varying A
**Prediction Trend Seen:** Feature_516 shows a distinct threshold effect around value 0.5, where the pass probability jumps sharply from 0.014 to 0.022. Below this threshold, the feature maintains a relatively stable impact on predictions, but crossing this boundary triggers a significant 1.3% increase in pass probability, as shown in the individual impact plot.

### Varying B
**Prediction Trend Seen:** Feature_244 demonstrates a similar threshold behavior at value 0.04, though with a smaller overall effect range of 0.7%. The transition from low to high probability is more abrupt than feature_516, with the pass probability jumping from 0.014 to 0.021 at the threshold point.

<img alt="feature_interaction_heatmap.png" src="evaluation%2Fexperiment_results%2Ffeature_interaction_heatmap.png" height="400"/>

### Varying A and B together
**Prediction Trend Seen:** The combined impact of both features shows a stronger effect than either feature alone, with the ability to swing predictions by 2.3%. The heatmap reveals that peak pass probability (0.033) occurs when both features are in their high ranges, suggesting synergistic behavior with a 1.0% additional interaction effect beyond their individual contributions.

### Varying A and B inversely
**Prediction Trend Seen:** The heatmap visualization shows distinct quadrants of prediction behavior, with the lowest pass probabilities (0.009) occurring when feature_516 is high and feature_244 is low. The upper-left and lower-right quadrants of the heatmap demonstrate how opposing variations in these features create regions of minimal prediction confidence, suggesting these features provide complementary rather than redundant information.