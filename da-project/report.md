# Heart Disease Prediction and Analysis

## Team

Team Member 1: [Your Name]
Team Member 2: [Your Name]

## Project Idea

This project aims to analyze the Cleveland Heart Disease Dataset to identify key risk factors associated with heart disease and develop a machine learning model capable of predicting heart disease presence. By exploring the relationships between clinical measurements and disease outcomes, we can better understand which factors contribute most significantly to heart disease diagnosis.

## Dataset

- **Source:** Cleveland Heart Disease Dataset (UCI Machine Learning Repository)
- **Size:** 303 patients, 14 features
- **Target variable:** Presence of heart disease (binary: 0 = no heart disease, 1 = heart disease)
- **Features:** age, sex, cp (chest pain type), trestbps (resting blood pressure), chol (cholesterol), fbs (fasting blood sugar), restecg (resting ECG), thalach (max heart rate), exang (exercise-induced angina), oldpeak (ST depression), slope, ca (major vessels), thal

## Python Analysis

Performed exploratory data analysis answering 10 clinical questions:

1. **Heart Disease Distribution:** 54% of patients have heart disease (164 out of 303)
2. **Age Distribution:** Patients with heart disease tend to be older (average ~57 years vs ~53 years)
3. **Gender Impact:** Males show higher prevalence of heart disease
4. **Chest Pain Types:** Type 4 (asymptomatic) shows strongest association with heart disease
5. **Blood Pressure:** Higher resting blood pressure correlates with increased risk
6. **Cholesterol:** Elevated cholesterol levels are associated with heart disease
7. **Max Heart Rate:** Lower maximum heart rate achieved is linked to heart disease
8. **Exercise Angina:** Exercise-induced angina is a strong predictor of heart disease
9. **Oldpeak (ST Depression):** Higher ST depression indicates greater heart disease risk
10. **Number of Vessels:** More major vessels colored by fluoroscopy correlates with heart disease

**Key findings from EDA:**
- Chest pain type (cp), maximum heart rate (thalach), number of major vessels (ca), and ST depression (oldpeak) have the strongest correlation with heart disease diagnosis
- The correlation heatmap reveals that these clinical indicators are most predictive of disease presence
- Age and sex are important demographic factors influencing heart disease risk

### Machine Learning: Random Forest Classifier

- **Accuracy:** ~85% on the test set
- **Model:** Random Forest with 100 estimators
- **Most Important Features:** ca (number of major vessels), oldpeak (ST depression), thalach (max heart rate), cp (chest pain type), and age

## R Analysis

Replicated key visualizations using ggplot2 with 5 main plots:

1. **Correlation Matrix (corrplot):** Shows all feature correlations with color-coded visualization
2. **Heart Disease Distribution:** Bar chart showing 164 patients with disease vs 139 without
3. **Age Distribution Histogram:** Age distribution colored by disease status
4. **Max Heart Rate Boxplot:** Compares thalach values between disease and no-disease groups
5. **Chest Pain Type Bar Chart:** Grouped bar chart showing chest pain types by disease status

**Key visual insights from R plots:**
- The correlation matrix reveals that cp, thalach, ca, and oldpeak have the strongest positive correlation with heart disease
- Patients with heart disease tend to have lower maximum heart rates (mean ~139 bpm vs ~158 bpm for healthy patients)
- Chest pain type 4 (asymptomatic) is most strongly associated with heart disease diagnosis

## Conclusion

The analysis revealed several key insights about heart disease risk factors. The most predictive features are the number of major vessels (ca), ST depression (oldpeak), maximum heart rate (thalach), and chest pain type (cp). The Random Forest model achieved approximately 85% accuracy, demonstrating that machine learning can effectively predict heart disease from clinical measurements. These findings align with medical knowledge - patients with exercise-induced symptoms, lower exercise capacity, and more blocked vessels are at higher risk for heart disease. This analysis provides a foundation for early heart disease detection and risk assessment using non-invasive clinical measurements.