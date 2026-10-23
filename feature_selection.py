import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.compose import ColumnTransformer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
fs=pd.read_excel(r"customer_churn_update.xlsx")
print(fs)

x = fs.drop(["Churn", "CustomerID", "JoinDate"], axis=1)

selected_features = [
    "TenureMonths",
    "TotalCharges",
    "SatisfactionScore",
    "MonthlyCharges",
    "City",
    "DataUsageGB",
    "LastLoginDays",
    "LoginFrequency",
    "Age",
    "PaymentMethod"
]

x_selected = fs[selected_features]

y=fs["Churn"].map({"No":0,"Yes":1})
# features selections  columns 
x_selected = fs[selected_features]

num_selected = x_selected.select_dtypes(
    include=["int64", "float64"]
).columns

cat_selected = x_selected.select_dtypes(
    include=["str"]
).columns

preprocessing_selected = ColumnTransformer([
    ("num", StandardScaler(), num_selected),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_selected)
])


# data base real columns
num=x.select_dtypes(include=["int64","float64"]).columns  # yahapar churn column ko drop karke 
#baki bachi column he or usko yaha par acess kiya he
cat=x.select_dtypes(include=['str']).columns

preprocessing=ColumnTransformer([
    ("num",StandardScaler(),num),
    ("cat",OneHotEncoder(handle_unknown='ignore'),cat)
])

model_random=Pipeline([

    ("preprocessing",preprocessing),
    ("random",RandomForestClassifier(class_weight='balanced',random_state=10))
])
#new features lane par pura piplie or columntransform change matlab sab new karna padta he  ho gata he 

model_random_2=Pipeline([

    ("preprocessing",preprocessing_selected),
    ("random",RandomForestClassifier(class_weight='balanced',random_state=10))
])

logistic_model=Pipeline([
       ("preprocessing",preprocessing_selected),
        ("logistic",LogisticRegression(class_weight='balanced'))
])

# hyper parametre for logistice regression because of logistic is very high f1 score in compare to others model  

param_grid = {
    "logistic__C": [0.01, 0.1, 1, 10],
    "logistic__solver": ["liblinear", "lbfgs"]
}

hypermeter_logistic=GridSearchCV(
    estimator=logistic_model,
    param_grid=param_grid,
    scoring='f1',
    n_jobs=-1,
    cv=5
)
decision_model=Pipeline([
       ("preprocessing",preprocessing_selected),
        ("decision",DecisionTreeClassifier(class_weight='balanced'))
])


x_selected_train,x_selected_test,y_train,y_test=train_test_split(
 
x_selected,y,test_size=0.2,random_state=10
)

model_random.fit(x,y)

importance = model_random.named_steps["random"].feature_importances_

features_name=model_random["preprocessing"].get_feature_names_out()


for feature,score in zip(features_name,importance):
    print(feature,":",score)


importance_dict = defaultdict(float)

for feature, score in zip(features_name, importance):

    if "__" in feature:
        original_feature = feature.split("__")[1]
    else:
        original_feature = feature

    if original_feature.startswith("Gender_"):
        original_feature = "Gender"

    elif original_feature.startswith("City_"):
        original_feature = "City"

    elif original_feature.startswith("ContractType_"):
        original_feature = "ContractType"

    elif original_feature.startswith("PaymentMethod_"):
        original_feature = "PaymentMethod"

    elif original_feature.startswith("InternetService_"):
        original_feature = "InternetService"

    elif original_feature.startswith("DeviceProtection_"):
        original_feature = "DeviceProtection"

    elif original_feature.startswith("TechSupport_"):
        original_feature = "TechSupport"

    elif original_feature.startswith("StreamingService_"):
        original_feature = "StreamingService"

    importance_dict[original_feature] += score


print("\nOriginal Column Wise Importance:")

for feature, score in sorted(
    importance_dict.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(feature, ":", score)


logistic_model.fit(x_selected_train,y_train)
y_predict_logistic=logistic_model.predict(x_selected_test)

accurancy= accuracy_score(y_test,y_predict_logistic)
print("accurancy:",accurancy)
precision=precision_score(y_test,y_predict_logistic)
print("precision_score:",precision)
recall=recall_score(y_test,y_predict_logistic)
print("recall_score:",recall)
f1=f1_score(y_test,y_predict_logistic)
print("f1_score:",f1)
print("confusion:",confusion_matrix(y_test,y_predict_logistic))
print("classification_report:",classification_report(y_test,y_predict_logistic))


decision_model.fit(x_selected_train,y_train)
y_predict_decision=decision_model.predict(x_selected_test)

accurancy= accuracy_score(y_test,y_predict_decision)
print("accurancy:",accurancy)
precision=precision_score(y_test,y_predict_decision)
print("precision_score:",precision)
recall=recall_score(y_test,y_predict_decision)
print("recall_score:",recall)
f1=f1_score(y_test,y_predict_decision)
print("f1_score:",f1)
print("confusion:",confusion_matrix(y_test,y_predict_decision))
print("classification_report:",classification_report(y_test,y_predict_decision))

model_random_2.fit(x_selected_train,y_train)
y_predict_random=model_random_2.predict(x_selected_test)

accurancy= accuracy_score(y_test,y_predict_random)
print("accurancy:",accurancy)
precision=precision_score(y_test,y_predict_random)
print("precision_score:",precision)
recall=recall_score(y_test,y_predict_random)
print("recall_score:",recall)
f1=f1_score(y_test,y_predict_random)
print("f1_score:",f1)
print("confusion:",confusion_matrix(y_test,y_predict_random))
print("classification_report:",classification_report(y_test,y_predict_random))

hypermeter_logistic.fit(x_selected_train,y_train)

print("best_meter:",hypermeter_logistic.best_params_)

print("best_scoring:",hypermeter_logistic.best_score_)


best_estimator=hypermeter_logistic.best_estimator_

# yaha se best estimator variable ko  predict kiya jata he 
y_predict_hyper=best_estimator.predict(x_selected_test)

print("Accuracy:", accuracy_score(y_test, y_predict_hyper))
print("Precision:", precision_score(y_test, y_predict_hyper))
print("Recall:", recall_score(y_test, y_predict_hyper))
print("F1 Score:", f1_score(y_test, y_predict_hyper))
print("confusion:",confusion_matrix(y_test,y_predict_hyper))
print("classification_report:",classification_report(y_test,y_predict_hyper))

#final model testing
new_customer = pd.DataFrame({
    "TenureMonths": [12],
    "TotalCharges": [4000],
    "SatisfactionScore": [10],
    "MonthlyCharges": [450],
    "City": ["Surat"],
    "DataUsageGB": [20],
    "LastLoginDays": [0],
    "LoginFrequency": [15],
    "Age": [25],
    "PaymentMethod": ["UPI"]
})
new_predict=best_estimator.predict(new_customer)
print(new_predict)
if new_predict==[1]:
    print("customer will churn")
else:
    print(" customer will not churn")


#best model ko save karne ki trik

best_estimator = hypermeter_logistic.best_estimator_
joblib.dump(best_estimator, "customer_churn_model.pkl")
print("model save succesfully!")