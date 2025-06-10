import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# 1. Load Dataset
water = pd.read_csv("water_potability.csv")

# 2. Handle Missing Values
water.fillna(water.mean(), inplace=True)

# 3. Feature & Target Separation
X = water.iloc[:, :-1]
y = water.iloc[:, -1]

# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply SMOTE to Balance Classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# 6. Train-Test Split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=0)

# 7. GridSearchCV for Hyperparameter Tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, cv=3, verbose=1)
grid.fit(Xtrain, ytrain)

# 8. Best Model from Grid Search
best_model = grid.best_estimator_

# 9. Predict & Evaluate
ytest_pred = best_model.predict(Xtest)

print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(ytest, ytest_pred))
print("Confusion Matrix:\n", confusion_matrix(ytest, ytest_pred))
print("Classification Report:\n", classification_report(ytest, ytest_pred))

# 10. Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(water.columns[:-1], best_model.feature_importances_)
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.grid()
plt.show()
