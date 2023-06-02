
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



veriler= pd.read_csv("auto-mpg.data")
columns= ["MPG","Silindir","Motor_Hacmi,","Beygir_Gucu","Ağırlık","İvme","Model_Yılı","Menşei"]
# MPG="Yakit Tüketim Ölçüsü"

data= pd.read_csv("auto-mpg.data",names = columns, comment = "\t",sep = " ", 
                     skipinitialspace = True,na_values = "?")
dataset=data.copy()

dataset.info()
print(dataset.describe())

print(dataset.isna().sum())

dataset["Beygir_Gucu"] = dataset["Beygir_Gucu"].fillna(dataset["Beygir_Gucu"].mean())
print(dataset.isna().sum())



corr_matrix = dataset.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation btw features")

sns.pairplot(dataset)
plt.show()


plt.figure()
sns.countplot(dataset["Menşei"])
print(dataset["Menşei"].value_counts())

plt.figure()
sns.countplot(dataset["Silindir"])
print(dataset["Silindir"].value_counts())

# one hot encoding

#  1=Amerika 2=Avrupa 3=Asya
dataset["Silindir"] = dataset["Silindir"].astype(str)  
dataset['Menşei'] = dataset['Menşei'].map({1: 'Amerika', 2: 'Avrupa', 3: 'Asya'})
dataset = pd.get_dummies(dataset, columns=['Menşei'], prefix='', prefix_sep='')
dataset.tail()
dataset = pd.get_dummies(dataset)


# Veriyi eğitim ve test verisi olarak ayırıyoruz

# Split
# drop bir sütunu tablodan kaldırmaya yarar
x = dataset.drop(["MPG"], axis = 1)
y = dataset.MPG
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)



# Standardization
 # RobustScaler #StandardScaler
 
scaler = RobustScaler()  # RobustScaler #StandardScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Linear regression

lm = LinearRegression()
model = lm.fit(X_train, Y_train)
print("LR Coef: ",lm.coef_)
 
rmse = np.sqrt(mean_squared_error(Y_train, model.predict(X_train)))
print(rmse) #eğitim hatası

rmse_test = np.sqrt(mean_squared_error(Y_test, model.predict(X_test)))
print(rmse_test) #test hatası

mse = mean_squared_error(Y_test, model.predict(X_test))
print("Linear Regression MSE: ",mse)


score=model.score(X_train,Y_train)


#Lasso Regression

lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)                                   


tuned_parameters = [{'alpha': alphas}]
n_folds = 5
gscv_lasso = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error',refit=True)
gscv_lasso.fit(X_train,Y_train)
scores = gscv_lasso.cv_results_['mean_test_score']
scores_std = gscv_lasso.cv_results_['std_test_score']
print("Lasso Coef: ",gscv_lasso.best_estimator_.coef_)
lasso = gscv_lasso.best_estimator_
print("Lasso Best Estimator: ",lasso)

mse = mean_squared_error(Y_test,gscv_lasso.predict(X_test))
print("Lasso MSE: ",mse)
print("---------------------------------------------------------------")
plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")

rmse_test_lasso = np.sqrt(mean_squared_error(Y_test,gscv_lasso.predict(X_test)))
rmse_test_lasso #test hatası

import xgboost as xgb

#XGBoost

parametersGrid = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000]}

model_xgb = xgb.XGBRegressor()

gscv = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs = 5, verbose=True)

gscv.fit(X_train, Y_train)
model_xgb = gscv.best_estimator_

mse = mean_squared_error(Y_test,gscv.predict(X_test))
print("XGBRegressor MSE: ",mse)

rmse_test_xgb = np.sqrt(mean_squared_error(Y_test, gscv.predict(X_test)))

rmse_test_xgb #test hatası


    