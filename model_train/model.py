import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import joblib

df=pd.read_csv("data\\data.csv")
df=df.drop_duplicates()
df=df.dropna()
X=df.drop("sign",axis=1)
y=df["sign"]
encoder=LabelEncoder()
y=encoder.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

params={
"n_estimators":[50,100,150,200,300],
"max_depth":[None,5,10,15,20],
"min_samples_split":[2,5,10],
"min_samples_leaf":[1,2,4],
"bootstrap":[True,False]
}

rf=RandomForestClassifier(random_state=42)

search=RandomizedSearchCV(
estimator=rf,
param_distributions=params,
n_iter=20,
cv=5,
scoring="accuracy",
n_jobs=-1,
random_state=42
)

search.fit(x_train,y_train)

model=search.best_estimator_

pred=model.predict(x_test)

print("best params:",search.best_params_)
print("cross val:",search.best_score_)
print("test accuracy:",accuracy_score(y_test,pred))
print()
print("classes:",list(encoder.classes_))
print()
print(classification_report(y_test,pred,target_names=encoder.classes_))
print()
print(confusion_matrix(y_test,pred))
joblib.dump(model,"objects\\model.pkl")
joblib.dump(encoder,"objects\\label_encoder.pkl")

print()
print("saved")