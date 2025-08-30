import pandas as pd
import numpy as np
from scipy.stats import randint,uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report, roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

#Loading data
train_df=pd.read_csv('data/train.csv')
test_df=pd.read_csv('data/test.csv')

#print(train_df.isnull().sum())
#Missing Data in Age, Cabin, and Embarked and 1 Fare in test which might cause some problems

#Filling Age with median
train_df['Age']=train_df['Age'].fillna(train_df['Age'].median())
test_df['Age']=test_df['Age'].fillna(test_df['Age'].median())

#Dropping Cabin cuz too many missing
train_df=train_df.drop(columns=['Cabin'])
test_df=test_df.drop(columns=['Cabin'])

#Filling Embarked with mode
train_df['Embarked']=train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked']=test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

#Filling Fare in test set
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median())

#As Sex and Embarked are not numbers, we will convert it
le=LabelEncoder()
train_df['Sex']=le.fit_transform(train_df['Sex'])
test_df['Sex']=le.fit_transform(test_df['Sex'])

train_df['Embarked']=le.fit_transform(train_df['Embarked'])
test_df['Embarked']=le.fit_transform(test_df['Embarked'])

#Defining X and Y

X=train_df.drop(columns=['Survived','PassengerId','Name','Ticket'])
Y=train_df['Survived']

#Scaling the data
scaler=StandardScaler()

X_trai,X_va,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=42)

X_train=scaler.fit_transform(X_trai)
X_val=scaler.transform(X_va)
#First Model Using Logistic Regression
log_reg=LogisticRegression(max_iter=500)
log_reg.fit(X_train,Y_train)

log_reg_y_pred=log_reg.predict(X_val)

print("Logistic Regression Accuracy: ",accuracy_score(Y_val,log_reg_y_pred))
print(confusion_matrix(Y_val,log_reg_y_pred))
print(classification_report(Y_val,log_reg_y_pred))


#Using Different Models to find out the best one as base
#SVC
svc=SVC(random_state=42)
svc.fit(X_train,Y_train)
svc_pred=svc.predict(X_val)

#Gradient Boosting
gb=GradientBoostingClassifier(learning_rate=0.01,random_state=42)
gb.fit(X_train,Y_train)
gb_pred=gb.predict(X_val)

#Decision Tree
dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train,Y_train)
dt_pred=dt.predict(X_val)

#Random Forest
rf=RandomForestClassifier(random_state=42,n_jobs=-1)
rf.fit(X_train,Y_train)
rf_pred=rf.predict(X_val)

results=pd.DataFrame({
    'Model' : ['LR','SVC','GB','DT','RF'],
    'Accuracy_score':[accuracy_score(Y_val,log_reg_y_pred),accuracy_score(Y_val,svc_pred),
                      accuracy_score(Y_val,gb_pred),accuracy_score(Y_val,dt_pred),
                      accuracy_score(Y_val,rf_pred)]
})
results=results.set_index('Model')
print(results)


'''Model     Accuracy           
LR           0.804469
SVC          0.815642
GB           0.798883
DT           0.782123
RF           0.82122'''
#Tuning the hyperparamters for the best 2

#Random Forest Parameter Grid
rf_grid={
    'n_estimators': randint(50,200),
    'max_depth': [None,10,20,30],
    'min_samples_split':randint(2,6),
    'min_samples_leaf':randint(1,4)
}

#SVC Parameter Grid
svc_grid={
    'max_iter':randint(500,2000),
    'degree':randint(1,6),
    'shrinking':[True,False],
    'cache_size':randint(100,300)
}

#Using RandomizedSearch
rf_rand=RandomizedSearchCV(
    RandomForestClassifier(),
    rf_grid,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42
)
rf_rand.fit(X_train,Y_train)

svc_rand=RandomizedSearchCV(
    SVC(),
    svc_grid,
    cv=5,
    n_iter=20,
    n_jobs=-1,
    random_state=42,
    scoring='accuracy'
)

svc_rand.fit(X_train,Y_train)

print("Random Forest Best Parameters: ",rf_rand.best_params_)
print('SVC Best Paramters: ',svc_rand.best_params_)

#Random Forest Best Parameters:  {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 103}
#SVC Best Paramters:  {'cache_size': 202, 'degree': 4, 'max_iter': 1360, 'shrinking': True}

#Using the final models

rf_final=RandomForestClassifier(max_depth=10, min_samples_leaf= 2, min_samples_split= 3, n_estimators=103,random_state=42)
svc_final=SVC(cache_size=202,degree=4,max_iter=1360,shrinking=True)

rf_final.fit(X_train,Y_train)
svc_final.fit(X_train,Y_train)

rf_final_pred=rf_final.predict(X_val)
svc_final_pred=svc_final.predict(X_val)

print("Accuracy of Random Forest: ", accuracy_score(Y_val,rf_final_pred))
print("Accuracy of SVC: ",accuracy_score(Y_val,svc_final_pred))

#Accuracy of Random Forest:  0.8324022346368715
#Accuracy of SVC:  0.8156424581005587

#Clearly Random Forest is the best model

X_test=test_df.drop(columns=['PassengerId','Name','Ticket'])

X_test_scaled=scaler.fit_transform(X_test)


test_pred=rf_final.predict(X_test_scaled)
print(np.unique(test_pred, return_counts=True))

results_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_pred
})

# Countplot
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=results_df, palette='pastel')
plt.title('Survival Predictions for Test Set')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Optional: Pie chart
survival_counts = results_df['Survived'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(survival_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title('Predicted Survival Distribution')
plt.show()