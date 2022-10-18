import yfinance as yf
import pandas as pd
import pandas_datareader as data
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble,metrics
from sklearn.metrics import classification_report,confusion_matrix
from xgboost.sklearn import XGBClassifier
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
import optuna
from sklearn import svm
yf.pdr_override()

data=pd.read_csv('機器學習_資料(股價)(訓練集).csv')
condition=data['2308 adjclose']>data['2308 adjclose'].shift(1)
data['2308 adjclose']=condition
data['2308 adjclose']=data['2308 adjclose'].astype(int)
data['2308 adjclose']=data['2308 adjclose'].shift(-1)
data=data.drop([2266])
data['2308 adjclose']=data['2308 adjclose'].astype(int)
print(data['MACD'])
print(data)
print(data.isnull().sum())
x=data.drop(['date','2308 adjclose'],axis=1).copy()
y=data['2308 adjclose']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 5)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train)
print(y_train)
q=pd.DataFrame(y_train)
q.hist()
plt.title('y_train')
#plt.show()
from sklearn.svm import SVR
from sklearn.svm import SVC
def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['rbf', 'sigmoid'])
    c = trial.suggest_float("C", 0.1, 10.0)
    gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
    model=SVC(
        kernel=kernel,
        gamma=gamma,
        C=c,
        random_state=4,
        probability=True
    )
    model.fit(x_train,y_train)
    return 1.0 - metrics.f1_score(y_test, model.predict(x_test))


study=optuna.create_study()
study.optimize(objective,n_trials=100)
print(study.best_params)
print(1.0-study.best_value)

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
import plotly.express as plotly
plotly_config={"staticPlot": True}
fig=plot_optimization_history(study)
fig.show(config=plotly_config)
fig = plot_param_importances(study)
fig.show(config=plotly_config)

model=SVC(
    kernel=study.best_params['kernel'],
    gamma=study.best_params['gamma'],
    C=study.best_params['C'],
    random_state=4,
    probability=True

)


model.fit(x_train,y_train)
y_hat=model.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_hat)
f1_score=metrics.f1_score(y_test,y_hat)
recall=metrics.recall_score(y_test,y_hat)
precision=metrics.precision_score(y_test,y_hat)
print(f'Accuracy:{accuracy:.5f}/ F1 Score:{f1_score:.5f}/Recall:{recall:.5f}/ Precision:{precision:.5f}')
#print('each f1:',CV5F_acc)
#print('Average f1:',round((np.mean(CV5F_acc))*100,2),'+/-',round((np.std(CV5F_acc))*100,2))
print('****svm***')
plot_confusion_matrix(y_test,y_hat,cmap='Accent')
#plt.show()


test_data=pd.read_csv('機器學習_資料(測試集)(股價).csv')
predict_result=model.predict_proba(test_data)
#print(predict_result)
predict_result_df=pd.DataFrame(predict_result)
#print(predict_result_df)
predict_result_df.to_csv('機器學習輸出機率(svm)(未標準化).csv')

plt.show()
