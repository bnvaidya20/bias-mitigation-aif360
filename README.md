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

### 4. Definition of Privileged and Unprivileged Groups

For both gender and ethnicity, we defined the groups as follows:

- **Gender**:
  - **Privileged Group (Male)**: `{'gender': 1}`
  - **Unprivileged Group (Female)**: `{'gender': 0}`

- **Ethnicity**:
  - **Privileged Group (White)**: `{privileged_ethnicity: 1}` where `privileged_ethnicity = 'ethnicity_WHITE'`
  - **Unprivileged Group (Non-White)**: `{unprivileged_ethnicity: 1}` where `unprivileged_ethnicity = 'ethnicity_NON-WHITE'`

These definitions were used to evaluate fairness metrics and assess bias in the models.

## Explanatory Analysis of Gender and Ethnicity Bias Mitigation Evaluation

### Gender Bias Evaluation

#### 1. Initial Evaluation

**Random Forest Model:**
- **Accuracy**: 0.71
- **Classification Report**: 
  - The model completely failed to predict the minority class (0), with precision, recall, and F1-score all being 0. This indicates that the model is heavily biased towards predicting the majority class (1).
  - For class 1, the recall is perfect (1.00), but the precision is lower, leading to an overall accuracy of 0.71.
  
**XGBoost Model:**
- **Accuracy**: 0.79
- **Classification Report**:
  - The model performed significantly better on class 0, with a precision of 1.00 but a recall of 0.25. This suggests that while the model is confident when it predicts class 0, it often fails to do so.
  - Class 1 shows a strong recall (1.00) and good precision (0.77), indicating that the model is more balanced compared to the Random Forest.

**Voting Classifier Model:**
- **Accuracy**: 0.75
- **Classification Report**:
  - The model has very high precision for class 0 (1.00) but a recall of only 0.12, meaning it rarely predicts class 0 correctly.
  - The performance for class 1 is strong, with perfect recall and good precision.

- **Fairness Analysis**:
  - **Difference in Mean Outcomes**: -0.108
    - This negative value indicates that the unprivileged group (female) has slightly lower predicted outcomes compared to the privileged group (male).
  - **Disparate Impact**: 1.154
    - This value is close to 1, suggesting no significant bias in the model's predictions regarding gender. A Disparate Impact within the range of 0.8 to 1.25 is generally considered acceptable.
  - **Other Fairness Metrics**: 
    - The Demographic Parity Difference, Equal Opportunity Difference, and Average Odds Difference are all close to 0, further supporting the conclusion that there is no significant bias.
    - The Theil Index, a measure of inequality, is low (0.050), indicating a fairly equitable distribution of predictions across genders.

#### 2. Evaluation with Bias Mitigation

**Random Forest Model:**
- **Accuracy**: 0.71 (unchanged)
- **Classification Report**: No change from the initial evaluation, indicating that bias mitigation did not affect the model's performance.

**XGBoost Model:**
- **Accuracy**: 0.75 (slightly lower than initial)
- **Classification Report**:
  - The model's performance on class 0 slightly improved, with recall increasing from 0.25 to 0.36. However, the precision for class 1 dropped slightly, leading to a small decrease in overall accuracy.

**Voting Classifier Model:**
- **Accuracy**: 0.71 (unchanged)
- **Classification Report**: No significant change from the initial evaluation.

- **Fairness Analysis**:
  - **Fairness Metrics**:
    - **Demographic Parity Difference**: Improved to 0.056, indicating a slight increase in parity between genders.
    - **Equal Opportunity Difference**: Increased to 0.100, suggesting a slight improvement in equal opportunity across genders.
    - **Average Odds Difference**: Decreased slightly to -0.017, indicating a minor improvement.
    - **Disparate Impact**: Improved to 1.065, staying within the acceptable range, reinforcing the absence of significant bias.
    - **Theil Index**: Increased slightly to 0.088, indicating a small increase in inequality, though it remains low overall.

### Ethnicity Bias Evaluation

#### 1. Initial Evaluation

**Random Forest Model:**
- **Accuracy**: 0.71
- **Classification Report**:
  - As with the gender evaluation, the model failed to predict class 0, indicating bias towards predicting class 1.

**XGBoost Model:**
- **Accuracy**: 0.71
- **Classification Report**:
  - The model performed slightly better than Random Forest, with more balanced precision and recall across both classes, although it still struggled with class 0.

**Voting Classifier Model:**
- **Accuracy**: 0.71
- **Classification Report**: Similar to the Random Forest model, the Voting Classifier did not predict class 0 well.

- **Fairness Analysis**:
  - **Difference in Mean Outcomes**: -0.023
    - This small negative value indicates a slight bias against the unprivileged ethnicity group (non-white).
  - **Disparate Impact**: 0.917
    - This value is within the acceptable range, suggesting no significant bias in the model's predictions regarding ethnicity.
  - **Other Fairness Metrics**: 
    - The metrics, including Demographic Parity Difference and Equal Opportunity Difference, show minor disparities but nothing indicating significant bias.
    - The Theil Index (0.126) is slightly higher than for gender, indicating a bit more inequality in the distribution of predictions across ethnicities.

#### 2. Evaluation with Bias Mitigation

**Random Forest Model:**
- **Accuracy**: 0.71 (unchanged)
- **Classification Report**: No change, indicating that bias mitigation did not affect the model's performance.

**XGBoost Model:**
- **Accuracy**: 0.68 (slightly lower than initial)
- **Classification Report**:
  - The performance on class 0 improved slightly, but the overall accuracy decreased. This suggests that bias mitigation helped with balancing predictions between classes, though it came at a slight cost to overall performance.

**Voting Classifier Model:**
- **Accuracy**: 0.68 (slightly lower than initial)
- **Classification Report**: Similar to XGBoost, there was a small drop in accuracy, but the changes suggest more balanced predictions between classes.

- **Fairness Analysis**:
  - **Fairness Metrics**:
    - **Demographic Parity Difference**: Improved to 0.015, indicating a small increase in parity between ethnicities.
    - **Equal Opportunity Difference**: Increased to 0.188, suggesting more disparity in equal opportunity across ethnicities after bias mitigation.
    - **Average Odds Difference**: Decreased slightly to -0.073, indicating minor improvement.
    - **Disparate Impact**: Improved to 1.019, remaining within the acceptable range and indicating that the bias is further reduced.
    - **Theil Index**: Increased to 0.167, suggesting a slight increase in inequality, but it remains at an acceptable level.

## Conclusion

This project highlights the complexities of balancing predictive accuracy and fairness in machine learning models. While bias mitigation techniques can improve fairness, they may also impact overall model performance. Our analysis shows that models like XGBoost can achieve a better balance between fairness and accuracy, making them a strong choice for applications where both are critical. However, further optimization and alternative strategies may be necessary to address persistent biases effectively.