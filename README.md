# Bias Mitigation in Machine Learning Models Using AI Fairness 360

## Introduction

In this project, we explore bias mitigation techniques in machine learning models, focusing on both gender and ethnicity biases. The goal is to assess how different models, including Random Forest, XGBoost, and Voting Classifier, perform in terms of predictive accuracy and fairness across different demographic groups. By applying bias mitigation strategies, we aim to improve the fairness of these models without compromising their performance.

## Problem Statement

Machine learning models are increasingly being used in critical decision-making processes. However, these models can inadvertently perpetuate or even exacerbate existing biases present in the data, leading to unfair outcomes for certain groups. This project addresses the following key questions:

1. How do popular machine learning models perform in terms of predictive accuracy and fairness across gender and ethnicity groups?
2. Can bias mitigation techniques improve the fairness of these models while maintaining or enhancing their predictive performance?
3. How do different models compare in terms of their ability to handle and mitigate bias?

## Dataset Used

The dataset used in this project is derived from the MIMIC-III Demo database, which contains clinical data from intensive care unit (ICU) patients. For this analysis, we focused on the following attributes:

- **Gender**: Binary attribute (Male, Female).
- **Ethnicity**: Categorical attribute with multiple classes (e.g., White, Black/African American, Asian, etc.), which were further aggregated into broader categories for analysis.
- **Survival**: The target variable indicating patient survival.

The dataset was split into training and test sets with attention to maintaining the proportion of different gender and ethnicity groups.

## Methodology

### 1. Initial Model Training

We trained three models—Random Forest, XGBoost, and a Voting Classifier—on the dataset to establish baseline performance metrics. The models were evaluated for both predictive accuracy and fairness, focusing on gender and ethnicity as protected attributes.

### 2. Bias Mitigation

To address potential biases, we applied bias mitigation techniques, particularly reweighing, to the training data. This technique adjusts the importance of each instance in the training set to reduce bias against unprivileged groups.

### 3. Evaluation of Bias Mitigation

After applying bias mitigation, the models were retrained and re-evaluated on the test set. We compared the performance metrics (e.g., accuracy, precision, recall) and fairness metrics (e.g., Disparate Impact, Demographic Parity Difference) before and after mitigation to assess the effectiveness of the bias mitigation strategies.

#### 3a. AI Fairness 360

AI Fairness 360 is extensible open source toolkit that allows to examine, report, and mitigate discrimination and bias in ML models throughout the AI application lifecycle. 

## Explanatory Analysis of Gender and Ethnicity Bias Mitigation Evaluation

### Gender Bias Evaluation

#### 1. Initial Evaluation

- **Random Forest Model**: The model showed significant bias towards predicting the majority class (1), with complete failure in predicting the minority class (0).
- **XGBoost Model**: Performed better than Random Forest with balanced predictions but still struggled with class 0.
- **Voting Classifier Model**: Similar to the Random Forest, it had high precision for class 0 but poor recall, indicating a bias towards class 1.

- **Fairness Metrics**:
  - The initial fairness analysis showed a slight bias against the unprivileged gender group, but overall, the Disparate Impact and other fairness metrics indicated no significant bias.

#### 2. Evaluation with Bias Mitigation

- **Random Forest Model**: No significant change after bias mitigation.
- **XGBoost Model**: Slight improvements in class 0 performance and fairness metrics, though with a small drop in overall accuracy.
- **Voting Classifier Model**: No significant change after bias mitigation.

- **Fairness Metrics**:
  - Bias mitigation slightly improved fairness metrics, particularly for the XGBoost model, suggesting a more balanced treatment of gender groups.

### Ethnicity Bias Evaluation

#### 1. Initial Evaluation

- **Random Forest Model**: Similar to the gender analysis, the model struggled with predicting the minority class (0).
- **XGBoost Model**: Showed better balance than Random Forest, though it still faced challenges with class 0 predictions.
- **Voting Classifier Model**: Again, poor performance in predicting class 0, indicating a bias towards class 1.

- **Fairness Metrics**:
  - The initial analysis indicated no significant bias, with fairness metrics within acceptable ranges, although there was a slight bias against the unprivileged ethnicity group.

#### 2. Evaluation with Bias Mitigation

- **Random Forest Model**: No significant change after bias mitigation.
- **XGBoost Model**: Showed improved balance in class predictions at the cost of a small drop in overall accuracy.
- **Voting Classifier Model**: Similar to XGBoost, with slightly better fairness metrics but lower accuracy.

- **Fairness Metrics**:
  - Bias mitigation resulted in a minor reduction in bias, particularly for the XGBoost and Voting Classifier models, with a slight increase in fairness metrics like Disparate Impact.

## Conclusion

This project highlights the complexities of balancing predictive accuracy and fairness in machine learning models. While bias mitigation techniques can improve fairness, they may also impact overall model performance. Our analysis shows that models like XGBoost can achieve a better balance between fairness and accuracy, making them a strong choice for applications where both are critical. However, further optimization and alternative strategies may be needed to address persistent biases effectively.
