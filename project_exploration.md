#### SER494: Exploratory Data Munging and Visualization
#### Semiconductor Yield Prediction
#### Peter Fischbach
#### 10/22/24

## Basic Questions
**Dataset Author(s):** Michael McCann, Adrian Johnston

**Dataset Construction Date:** 11/18/2008

**Dataset Record Count:** 1567

**Dataset Field Meanings:** Pass/Fail | Timestamp | ... Signals/Variables (Unknown)

**Dataset File Hash(es):**
 - secom_labels.data: ec589b71d8fd35adef98bca758bcc545
 - secom.data: e4716cda08f7e67afe99a199eec6a801
 - secom.names: 5de417d5281109d9d9f79634e87942ab

## Interpretable Records
### Record 1
**Raw Data:** -1 "19/07/2008 21:35:00" ...Signals/Variables

**Interpretation:** The record from 19/07/2008 at 9:35 pm passed in house line testing with the associated signal/variable values

### Record 2
**Raw Data:** 1 "19/07/2008 21:57:00" ...Signals/Variables

**Interpretation:** The record from 19/07/2008 at 9:57 pm failed in house line testing with the associated signal/variable values

## Background Domain Knowledge

Semiconductor manufacturing yield is a critical metric that directly impacts profitability and competitiveness in the semiconductor industry. As explained by Samsung Electronics, "semiconductor yield is a percentage of the total number of chips that were actually produced to the maximum chip (IC) count on one wafer" and represents the opposite of the defective rate [1]. The importance of yield cannot be overstated - a single percentage point improvement can translate to millions of dollars in additional revenue for semiconductor companies.

The complexity of semiconductor manufacturing means yield is affected by numerous interrelated factors. According to Ramiro Palma et al. in a report, process complexity has grown exponentially with modern chips containing over 100 billion transistors, making yield management increasingly challenging [2]. Each of the hundreds of manufacturing steps has the potential to introduce defects through various mechanisms. Even minor variations in temperature, pressure, or particle contamination levels can significantly impact the final yield.

The semiconductor manufacturing process requires incredible precision and cleanliness. As described in an ASML article, modern fabs must maintain extremely strict cleanroom conditions where even microscopic particles can cause devastating defects [3]. A class 1 cleanroom, typical for semiconductor manufacturing, allows only one 0.5 micron particle per cubic foot of air - for comparison, a human hair is about 75 microns wide. This level of cleanliness is essential because the features on modern chips can be just a few nanometers in size.

Environmental control plays a crucial role in yield management. Temperature, humidity, vibration levels, and electromagnetic interference must all be precisely controlled. Additionally, the chemical purity of materials used in the process must meet extremely strict specifications. Any deviation from these parameters can result in defects that impact yield.

The industry typically divides yield into two main components: line yield and die yield. Line yield measures the percentage of wafers that successfully complete the entire manufacturing process without being scrapped. Die yield represents the percentage of good dies on completed wafers. Both metrics must be optimized to achieve profitable manufacturing operations.

Understanding and optimizing yield requires deep expertise across multiple disciplines. As the industry continues to push the boundaries of semiconductor technology, maintaining high yields becomes increasingly challenging but remains essential for competitive manufacturing operations. With wafer costs exceeding $20,000 for leading-edge processes, yield optimization directly impacts a company's bottom line and ability to compete in the market.

[Source [1]](https://semiconductor.samsung.com/us/support/tools-resources/dictionary/semiconductor-glossary-yield/)

[Source [2]](https://www.semiconductors.org/wp-content/uploads/2022/11/2022_The-Growing-Challenge-of-Semiconductor-Design-Leadership_FINAL.pdf)

[Source [3]](https://www.asml.com/en/technology/all-about-microchips/how-microchips-are-made)

## Dataset Generality
The distribution of my dataset is representative of real-world data from the semiconductor industry. Semiconductor yield data can have hundreds or even thousands of features based on different data from sensors at the end of the manufacturing process. My dataset contains 591 features and falls within the expected range for yield data. Since there are so many features, there needs to be a lot of records to effectively analyze the data. This leads to a problem where I only have 1567 instances and, even without doing the math, is clearly not enough for all scenarios. However, that is assuming that the features are not independent of each other. Luckily my data’s features are independent of each other since they are all independent sensors recording different information.


## Data Transformations
### Transformation 1
**Description:**  
Impute NaN values using record class' median and standard deviation.

**Soundness Justification:**  
This transformation is sound because it does not change the semantics of the data. This is done by calculating data based on the median for that records class (pass/fail). Usable data is not discarded, nothing is being discarded, only data is being added. Errors are not being introduced, I tested the outputs after calculating the missing values. Lastly, outliers are not being introduced since the missing values are based on the median and randomized within 1 standard deviation.


## Visualizations
### Visual 1
![Number of wafers Passed vs Failed.png](data_processed%2FNumber%20of%20wafers%20Passed%20vs%20Failed.png)
**Analysis:**  
This shows the distribution of my data based on whether the wafer passed or failed inline house testing. The results are within expected range where a good yield percent can be ~96% and this is ~93%.

### Visual 2
![feature_0 vs feature_1.png](data_processed%2Ffeature_0%20vs%20feature_1.png)
**Analysis:**  
This shows the correlation between feature_0 and feature_1 in my data. Using the chart, we can se there is no real correlation between the features. If anything there may be a small negative correlation between feature_0 and feature_1.

### Visual 3
![feature_2 vs feature_3.png](data_processed%2Ffeature_2%20vs%20feature_3.png)
**Analysis:**  
This shows the correlation between feature_2 and feature_3 in my data. The chart shows that there is a more significant positive correlation compared to the other scatter plots.

### Visual 4
![feature_1 vs feature_2.png](data_processed%2Ffeature_1%20vs%20feature_2.png)
**Analysis:**  
This shows the correlation between feature_1 and feature_2 in my data. The chart shows that there is no correlation between the data. The data is mostly centered with some outliers but there is no pattern to be seen.

### Visual 5
![feature_0 vs feature_3.png](data_processed%2Ffeature_0%20vs%20feature_3.png)
**Analysis:**  
This shows the correlation between feature_0 and feature_3 in my data. The chart looks a bit different from the others because of the bottom "limit" for feature_3. We can see that feature_3 is facing resistance dropping below ~900 and there can be some potential for prediction there.