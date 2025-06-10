import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Load dataset
water = pd.read_csv("water_potability.csv")

# Handle missing values
water.fillna(water.mean(), inplace=True)

# Split features and target
X = water.iloc[:, :-1]
y = water.iloc[:, -1]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Create and train SVM model (you can experiment with different kernels: 'linear', 'poly', 'rbf', 'sigmoid')
svm_model = SVC(kernel='rbf', class_weight='balanced')
svm_model.fit(Xtrain, ytrain)

# Predict
ytest_pred = svm_model.predict(Xtest)

# Evaluation
print("Accuracy:", accuracy_score(ytest, ytest_pred))
print("Confusion Matrix:\n", confusion_matrix(ytest, ytest_pred))
print("Classification Report:\n", classification_report(ytest, ytest_pred))

# Optional: Scatter plot for visualization (actual vs predicted)
plt.figure()
plt.scatter(ytest, ytest_pred, color='green', alpha=0.5)
plt.title('SVM: Actual vs Predicted Potability')
plt.xlabel('Actual Potability')
plt.ylabel('Predicted Potability')
plt.grid()
plt.show()

