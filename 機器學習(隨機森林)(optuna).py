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

def objective(trial):
    max_depth = trial.suggest_int('max_depth', 2, 128)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 128)
    max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 128))
    min_samples_leaf = int(trial.suggest_int('min_samples_leaf', 2, 128))
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_features = trial.suggest_categorical('max_features',['auto', 'sqrt'])
    bootstrap = trial.suggest_categorical('bootstrap',[True, False])
    RFC=RandomForestClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_split=min_samples_split,
        criterion=criterion,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=10
    )
    RFC.fit(x_train,y_train)
    return 1.0 - metrics.f1_score(y_test, RFC.predict(x_test))


study=optuna.create_study()
study.optimize(objective,n_trials=50)
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

model=RandomForestClassifier(
        max_depth=study.best_params['max_depth'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_leaf_nodes=study.best_params['max_leaf_nodes'],
        min_samples_split=study.best_params['min_samples_split'],
        criterion=study.best_params['criterion'],
        max_features=study.best_params['max_features'],
        bootstrap=study.best_params['bootstrap'],
        random_state=10

)

model.fit(x_train,y_train)
y_hat=model.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_hat)
f1_score=metrics.f1_score(y_test,y_hat)
recall=metrics.recall_score(y_test,y_hat)
precision=metrics.precision_score(y_test,y_hat)
print(f'Accuracy:{accuracy:.5f}/ F1 Score:{f1_score:.5f}/Recall:{recall:.5f}/ Precision:{precision:.5f}')
CV5F_acc=cross_val_score(model,x_train,y_train,cv=5,scoring='f1')
#print('each f1:',CV5F_acc)
#print('Average f1:',round((np.mean(CV5F_acc))*100,2),'+/-',round((np.std(CV5F_acc))*100,2))
print('**** DecisionTree***')
plot_confusion_matrix(y_test,y_hat,cmap='Accent')
plt.show()

feature_names=x.keys().tolist()
result=pd.DataFrame(
    {'feature':feature_names,
    'feature_importance':model.feature_importances_.tolist()
    }
)
result=result.sort_values(by=['feature_importance'],ascending=False).reset_index(drop=True)
print(result)
result10=result.iloc[:10]


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%f'%float(height),
                 ha='center', va='bottom')
plt.style.use('ggplot')
fig = plt.figure(figsize=(100, 6))
gini = plt.bar(result10.index, result10['feature_importance'], align='center')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.xticks(result10.index, result10['feature'],fontsize=10)
autolabel(gini)
plt.show()

test_data=pd.read_csv('機器學習_資料(測試集)(股價).csv')
predict_result=model.predict(test_data)
print(predict_result)
predict_result_df=pd.DataFrame(predict_result)
print(predict_result_df)
predict_result_df.to_csv('隨機森林(未標準化).csv')
