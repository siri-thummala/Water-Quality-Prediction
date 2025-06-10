#  Water Quality Prediction using Machine Learning
Machine Learning project for predicting water quality using chemical parameters — combines domain knowledge in chemical engineering with AI techniques.
This project uses machine learning algorithms to predict the quality of water based on its chemical properties. It combines domain knowledge from chemical engineering with AI techniques like Logistic Regression, SVM, and XGBoost to classify water as safe or unsafe.

---

##  Dataset

The dataset contains various physicochemical parameters of water such as:
- pH
- Hardness
- Solids
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity
- Potability (Target variable: 0 = Not potable, 1 = Potable)


##  Algorithms Used

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **XGBoost Classifier**

Each model was trained and evaluated on the dataset, with accuracy and confusion matrix used for performance comparison.
##  Data Imbalance Handling & Tuning

### SMOTE (Synthetic Minority Over-sampling Technique)

The dataset was imbalanced, with fewer samples labeled as *potable* water. To address this, **SMOTE** was applied to generate synthetic examples of the minority class. This helped improve model generalization and reduced bias toward the majority class.

---

## 📊 Results

| Model                | Accuracy  |
|---------------------|-----------|
| Logistic Regression | 63%    |
| SVM                 | 66%    |
| XGBoost             | 70%    |





