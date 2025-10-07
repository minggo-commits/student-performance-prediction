# Machine Learning Project Report - Muh. Arsan Akbar

## Project Domain

Education is a fundamental pillar in developing competent and highly competitive human resources, especially in facing the challenges of globalization and technological advancement. Quality education is key to producing productive individuals capable of making positive contributions to society. However, education systems in various countries, including Indonesia, still face major challenges, particularly regarding high dropout rates and low student retention rates (Tjandra et al., 2022). This issue not only hinders the development of the education sector but also has long-term impacts on the future of the younger generation.

One of the main causes of high dropout rates is low student academic performance. When students have difficulty understanding subject matter and do not achieve adequate learning outcomes, they are more vulnerable to losing motivation and eventually choosing to leave the formal education system (Gusnina et al., 2022). In this context, appropriate preventive strategies are needed to identify at-risk students and provide early intervention (Ismanto et al., 2022).

The application of technologies such as machine learning in education provides great opportunities to address these problems. One approach is to build a student academic performance prediction system based on historical data and individual characteristics. Academic performance itself is an indicator that reflects the extent to which students successfully achieve learning objectives in a particular field. By predicting academic performance, educators can provide guidance, additional resources, and interventions tailored to the needs of each student (Masangu et al., 2020).

Furthermore, good academic performance not only determines student success in the school environment but also becomes an important factor in readiness to enter the workforce. Students who excel academically generally have greater opportunities to obtain quality jobs and have better career paths (Adane et al., 2023). Therefore, improving student academic performance has broad impacts, both from the education side and their professional lives after graduation.

Based on this urgency, this project will develop a student academic performance prediction system using a machine learning approach, leveraging features such as math, reading, and writing scores, as well as sociodemographic variables such as gender, ethnicity, and parental education background. Through this approach, it is hoped that schools and educational institutions can proactively detect potential academic problems and provide more effective data-based solutions.

References:

[Tjandra, E., Kusumawardani, S. S., & Ferdiana, R. (2022). Student performance prediction in higher education: A comprehensive review. AIP Conference Proceedings. https://doi.org/10.1063/5.0080187](https://doi.org/10.1063/5.0080187)

[Gusnina, M., Wiharto, N., & Salamah, U. (2022). Student performance prediction in Sebelas Maret University based on the Random Forest algorithm. Ingénierie Des Systèmes D Information, 27(3), 495–501. https://doi.org/10.18280/isi.270317]( 
https://doi.org/10.18280/isi.270317)

[Ismanto, E., Ghani, H. A., Saleh, N. I. M., Amien, J. A., & Gunawan, R. (2022). Recent systematic review on student performance prediction using backpropagation algorithms. TELKOMNIKA (Telecommunication Computing Electronics and Control), 20(3), 597. https://doi.org/10.12928/telkomnika.v20i3.21963]( 
http://doi.org/10.12928/telkomnika.v20i3.21963)

[Adane, M. D., Deku, J. K., & Asare, E. K. (2023). Performance analysis of machine learning algorithms in prediction of student academic performance. Journal of Advances in Mathematics and Computer Science, 38(5), 74–86. https://doi.org/10.9734/jamcs/2023/v38i51762]( 
https://doi.org/10.9734/jamcs/2023/v38i51762)

## Business Understanding

Education is the main pillar of human resource development. However, serious challenges are still faced, especially regarding low student academic performance which can trigger school dropouts. Therefore, the use of machine learning-based predictive approaches in education becomes a strategic step to detect potential academic problems early.

### Problem Statements
- How to accurately predict student academic performance based on certain characteristic data?
- What features are most significant and influential in determining student academic performance?

### Goals
- Build a machine learning prediction model capable of accurately estimating student academic performance based on available features.
- Identify features that are most correlated and contribute significantly to student academic success.

### Solution statements
- Apply KNeighborsRegressor, RandomForestRegressor and AdaBoostRegressor algorithms to create student performance prediction models
- Create new features from existing features (feature engineering), namely combining average scores from three main assessment indicators to form a new label. This feature will later be used as a reference to assess the most significant features for student success.
- Calculate Mean Squared Error for each algorithm on train and test data to find the best model
    
## Data Understanding

The dataset used in this project is the Student Performance Prediction Dataset sourced from the [Kaggle](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics/data) platform.

**Data Volume:** This dataset consists of 1000 rows (student samples) and 8 columns (features).

**Data Condition:** Based on initial analysis, this dataset has the following conditions:
- **Missing Values:** There are no missing values in the dataset.
- **Outliers:** Based on visualization of numerical feature distributions, no extreme outliers were detected that could significantly disrupt analysis or modeling. Some extremely low values in math scores may exist, but are considered natural variations in student performance.

**Data Source Link:** [https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics/data](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics/data)

**Feature Descriptions:**
- `gender`: Represents the student's gender, with values `male` or `female`.
- `race/ethnicity`: Shows the student's racial or ethnic background. This category is divided into five groups: `group A`, `group B`, `group C`, `group D`, and `group E`.
- `parental level of education`: Indicates the highest level of education achieved by the student's parent or guardian. Values include: `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, and `master's degree`.
- `lunch`: Indicates the student's lunch subsidy status. Values are `standard` (full payment) or `free/reduced` (free or subsidized).
- `test preparation course`: Indicates whether the student has completed a test preparation course. Values are `completed` or `none`.
- `math score`: Student's score in the standard mathematics subject exam. Values are integers from 0 to 100.
- `reading score`: Student's score in the standard reading subject exam. Values are integers from 0 to 100.
- `writing score`: Student's score in the standard writing subject exam. Values are integers from 0 to 100.



### Exploratory data analysis - Univariate Analysis
### Categorical Features
### Gender Distribution
| Gender | Count | Percent |
|--------|-------|---------|
| Male   | 508   | 50.8%   |
| Female | 492   | 49.2%   |

The number of male students (50.8%) and female students (49.2%) is nearly balanced. This shows no significant bias in gender representation.

### Race/Ethnicity Distribution
| Race/Ethnicity | Count | Percent |
|----------------|-------|---------|
| Group C        | 323   | 32.3%   |
| Group D        | 257   | 25.7%   |
| Group B        | 198   | 19.8%   |
| Group E        | 143   | 14.3%   |
| Group A        | 79    | 7.9%    |

Most students come from Group C (32.3%), followed by Group D (25.7%) and Group B (19.8%). Group A and E are relatively fewer.

### Parental Level of Education Distribution
| Parental Level of Education | Count | Percent |
|-----------------------------|-------|---------|
| Some college                | 224   | 22.4%   |
| High school                 | 215   | 21.5%   |
| Associate's degree          | 204   | 20.4%   |
| Some high school            | 177   | 17.7%   |
| Bachelor's degree           | 105   | 10.5%   |
| Master's degree             | 75    | 7.5%    |

The majority of student parents have an education level of "some college" (22.4%), followed by "high school" (21.5%) and "associate's degree" (20.4%). Meanwhile, only a small portion of parents have a "master's degree" (7.5%).

### Lunch Distribution
| Lunch Type     | Count | Percent |
|----------------|-------|---------|
| Standard       | 660   | 66.0%   |
| Free/Reduced   | 340   | 34.0%   |

66.0% of students receive standard lunch, while 34.0% receive free or discounted lunch.

### Test Preparation Course Distribution
| Test Preparation Course | Count | Percent |
|-------------------------|-------|---------|
| None                    | 656   | 65.6%   |
| Completed               | 344   | 34.4%   |

65.6% of students did not take a test preparation course, while 34.4% did.

**Conclusion**

Based on the results of exploratory data analysis (EDA) on categorical variables, it can be concluded that the distribution of students by gender is quite balanced, with a proportion of 50.8% male and 49.2% female. From an ethnic background perspective, the majority of students come from group C (32.3%), followed by group D (25.7%) and group B (19.8%), while groups A and E only contribute 7.9% and 14.3% respectively. Parental education background shows that most come from families with a middle education level, such as "some college" (22.4%) and "high school" (21.5%). Only a few parents have a master's degree (7.5%), indicating that most students may not receive academic support from highly educated parents.

From an economic perspective, 66.0% of students receive standard lunch, while 34.0% receive free or subsidized lunch, which often serves as an indicator of lower socioeconomic conditions. In addition, only 34.4% of students have completed a test preparation course, while 65.6% have not taken such courses. This shows that most students may face limitations in access to additional academic preparation.

Overall, these results provide an overview that socioeconomic factors, parental education background, and access to additional learning facilities can be important factors affecting student academic performance. Further analysis is highly recommended to see how these variables correlate with academic test results such as math, reading, and writing scores, in order to gain a deeper and more comprehensive understanding.

### Numerical Features
![EDA Unvariate](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Unvariate%20Numerical.png)

Based on the results of univariate exploratory data analysis on numerical features namely math score, reading score, and writing score, it can be concluded that the distribution of these three scores tends to follow a normal distribution pattern, although slightly left-skewed, especially on math and writing scores. Most values are in the range of 60 to 80, indicating that the majority of students have fairly good academic performance. Reading scores show the most symmetrical distribution, with more high-value concentrations than the other two scores, indicating that students' reading ability is generally superior. Meanwhile, math scores have some low values that in reality cannot be considered outliers, but are not too significant. Overall, these three scores show a good and stable distribution.

**Conclusion**

*   The first histogram displays the distribution of math exam scores (math score). It can be seen that the score distribution tends to be unimodal and approaches a normal distribution, although there is slight left skewness (the distribution tail extends toward lower values). Most students obtain scores between 60 and 80, with the frequency peak around 65-70. There are some students with very low scores (below 40) and also some students with very high scores (above 90), but their numbers are relatively fewer compared to the middle score group.
*   The second histogram presents the distribution of reading exam scores (reading score). The reading score distribution appears closer to a normal distribution compared to math scores. The frequency peak is around 70-80, and most students obtain scores between 60 and 90. The reading score spread also appears slightly wider compared to math scores, indicating that the variation in reading performance among students may be greater. The number of students with very low scores (below 40) and very high scores (above 95) is also relatively few.
*   The third histogram illustrates the distribution of writing exam scores (writing score). The writing score distribution also appears unimodal with the frequency peak around 65-75. Most students obtain scores between 55 and 85. The writing score distribution shows more pronounced negative skewness compared to math scores, with a longer distribution tail toward lower values. This indicates that there are more students who obtain scores below average compared to students who obtain scores far above average.


### Exploratory data analysis - Multivariate Analysis
### Categorical Features Against Target
![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Gender.png)
Female students have a slightly higher average score (±70) compared to male students (±68). Because the difference is small, this feature has a low influence on the average score.

![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Lunch.png)
Students who receive standard lunch have a higher average score (±72) compared to students who receive free lunch (±64). This shows that lunch status has a fairly strong influence on the average score.

![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Pendidikan%20Orangtua.png)
Students with highly educated parents such as bachelor's degree and master's degree tend to have higher average scores (±71), while those from parents with some high school education have lower average scores (±65). Although there appears to be a trend, the differences between groups are not too sharp, so this feature does not have too much influence on scores.

![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Persiapan%20Tes.png)
Students who complete test preparation courses have a higher average score (±74) compared to those who do not take courses (±67). This shows that test preparation courses have a fairly strong influence on increasing average scores.

![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Ras.png)
Group E has the highest average score (±76), while other groups range from 67 to 71. This difference shows variation, but is not consistently increasing or decreasing between groups, so this feature has a low influence on scores.

**Conclusion**

The analysis results show that categorical variables such as parental education, gender, lunch type, parental education level, course participation, and race actually do not have a significant impact on students' average score values. This is indicated by the average score value for each variable which is only in the range of 60-75.

### Numerical Features Against Target
![EDA Multivariate Numerical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Numerical.png)
Numerical features show strong linear relationships with each other, with the highest correlation between reading score and writing score. Average score depends proportionally on the three original scores, and this linear relationship validates its use as a combined metric.

**Conclusion**

This pair plot analysis clearly shows very strong positive linear relationships between math, reading, and writing scores with students' average scores. In addition, there is a high correlation among the three exam scores themselves. This finding underlines the importance of these three subjects in determining students' overall academic performance. The average score prediction model will likely be very accurate if using these three exam scores as features. There are no significant non-linear patterns or extreme outliers that need special attention from this visualization.

## Data Preparation

- Since the dataset shows a clean condition without missing values and no significant outliers that could have a negative impact, no data reduction was performed.

- Feature Engineering: Creating a new feature in the form of an average score obtained from math score + reading score + writing score divided by 3, this was done considering there was no target in the dataset, so feature engineering was needed to generate a new relevant feature.

- Categorical feature encoding: Categorical feature encoding such as OneHotEncoder is important because most machine learning algorithms cannot handle categorical data directly. They require numerical input, while some features in the dataset are categorical, these features are gender, race, parental education level, lunch type, and test preparation.

- Dimensionality reduction with PCA: Dimensionality reduction with PCA (Principal Component Analysis) is needed because the math score, reading score, and writing score features show high correlation with each other, meaning there is information redundancy. PCA helps simplify these features into several main components that still retain most of the information, so it can improve model efficiency, reduce overfitting risk, and facilitate data visualization. In addition, PCA also helps eliminate noise and maintain data structure in lower dimensions. Therefore, the math score, reading score, and writing score features are included in the PCA process into a feature called student performance.

- Train and test split: Train-test split needs to be done to objectively evaluate model performance. By dividing the data into training data (train) and test data (test), we can train the model on one part of the data and test it on data that has never been seen before. This is important to assess the model's ability to generalize to new data and prevent overfitting, which is a condition where the model is too good at memorizing training data but poor at predicting new data. In this case, the data is divided into 90% for training and 10% for testing, providing enough data for learning while still leaving representative data for evaluation.

- Standardization: this needs to be done to equalize the scale of numerical features so that machine learning models can work optimally. Features like student performance may have different value ranges compared to other features, and this can cause the model to be biased toward features with large values. By standardization using StandardScaler, data is transformed to have a mean of 0 and standard deviation of 1, so all features are on a balanced scale. This is especially important for algorithms that are sensitive to data scale such as KNN, SVM, and linear regression. Standardization results show that the data has been centered around zero with uniform standard spread, ensuring the model training process becomes more stable and accurate.

## Modeling
At this stage, machine learning model development was carried out to predict students' average scores based on previously processed input features. Three regression algorithms were used, namely K-Nearest Neighbors (KNN), Random Forest Regressor, and AdaBoost Regressor.

- **K-Nearest Neighbors (KNN)**
  
K-Nearest Neighbors (KNN) is a non-parametric algorithm that works by comparing the distance between test data and all training data, then selecting the k nearest neighbors to make predictions. The prediction value for regression is determined from the average target value of those k nearest neighbors. The KNN model is used with the parameter n_neighbors=10 and other parameters at default values. The advantages of KNN are simple and do not require complex training processes. However, KNN is very sensitive to feature scales and less efficient on large datasets. This model produces an MSE (mean squared error) of 0.0137 (train) and 0.0113 (test).

- **Random Forest**

Random Forest is an ensemble learning algorithm that combines many decision trees to improve prediction accuracy. Each tree is trained on a randomly selected subset of data (bootstrap), and the final prediction result is taken from the average of all trees. Random Forest is used with n_estimators=50, max_depth=16, random_state=55, n_jobs=-1 and other parameters at default values. This algorithm can handle data with non-linear features and is not sensitive to feature scales. However, random forest requires large computational resources and is less interpretable. Evaluation results show the best performance compared to other models, with very small MSE of 0.000009 (train) and 0.000008 (test). This shows the model is very accurate in capturing data patterns.

- **AdaBoost Regressor**

AdaBoost (Adaptive Boosting) works by forming an ensemble model from a number of weak learners, usually small-sized decision trees. Each new model is built with focus on data misclassified by the previous model. The final prediction result is a weighted combination of all models. AdaBoost is used with learning_rate=0.05 and random_state=55 and other parameters at default values. This algorithm improves model accuracy by combining many simple predictors, however it is vulnerable to outlier data and noise. The results obtained are quite good, with MSE 0.0024 (train) and 0.0029 (test), but still inferior to Random Forest.

**Conclusion**

Based on the MSE evaluation results on training and test data, Random Forest was chosen as the best model because it produces the lowest error among all tested models. In addition, this model is also more stable and able to handle data complexity without experiencing overfitting.

## Evaluation
Since this project is a regression case, the evaluation metric used is Mean Squared Error (MSE). MSE measures the average squared difference between actual values (y_true) and predicted values (y_pred). The smaller the MSE value, the more accurate the model in making predictions. This metric is suitable to use because it gives a greater penalty to prediction errors that are far from the actual value. MSE is calculated using the following formula:

![MSE Formula](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/07/image-37.png)


The way MSE works is by calculating the difference between actual and predicted values for each data point, then squaring that difference so there are no negative values and giving a greater penalty to prediction errors that are far off. Then all the squared errors are summed and averaged.

MSE is effectively used in regression because it provides an understanding of how large the average model error is in squared units of the target. A lower MSE value indicates a better model in predicting the target.

**Model Evaluation Results**

Based on the evaluation results of the three models, the following results were obtained:
| Model        | Train MSE | Test MSE |
|--------------|-----------|----------|
| KNN          | 0.0137    | 0.0113   |
| RandomForest | 0.000009  | 0.000008 |
| Boosting     | 0.0024    | 0.0029   |


From the table above, it can be seen that Random Forest Regressor has the best performance with the smallest MSE value both on training and test data. This shows that this model is able to generalize very well, and has minimal overfitting.

**Prediction Evaluation**

To see the quality of predictions further, a comparison was made between actual values (y_true) and prediction results from the three models:
| y_true | KNN  | RandomForest | Boosting |
|--------|------|--------------|----------|
| 56.3   | 60.3 | 56.3         | 56.7     |
| 92.0   | 87.2 | 92.0         | 93.2     |
| 72.0   | 73.5 | 72.0         | 70.4     |
| 63.3   | 67.8 | 63.3         | 63.8     |

From that table, it can be seen that Random Forest prediction results are most consistently close to the actual values compared to other models. This strengthens the reason for choosing Random Forest as the final model.
