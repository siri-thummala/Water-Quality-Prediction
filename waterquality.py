import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#dataset
water=pd.read_csv("water_potability.csv")
water.fillna(water.mean(), inplace=True)
X=pd.DataFrame(water.iloc[:,0:9])
y=pd.DataFrame(water.iloc[:,9:10])
print(X.head())
print(y.head())

#preprocessing

scaler=StandardScaler()
X=scaler.fit_transform(X)

Xtrain,Xtest,ytrain,ytest=train_test_split(X, y,test_size=0.2,random_state=0)

reg = LogisticRegression()
reg.fit(Xtrain, ytrain.values.ravel())
LogisticRegression(class_weight='balanced')

ytestpredict = reg.predict(Xtest)
 


print("Accuracy:", accuracy_score(ytest, ytestpredict))
print("Confusion Matrix:\n", confusion_matrix(ytest, ytestpredict))
print("Classification Report:\n", classification_report(ytest, ytestpredict))

print(water['Potability'].value_counts())

plt.figure()
plt.scatter(ytest, ytestpredict, color='blue', alpha=0.6)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Values') 
plt.xlabel('Actual potability')
plt.ylabel('Predicted potability')
plt.grid()
plt.show()
