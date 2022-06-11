from turtle import pen
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
import seaborn as sns
# from learn import X_test, X_train; sns.set_theme(color_codes=True)
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def info_on_data(df):
    print("Head:")
    print(df.head())
    print(":::::::")
    print("Information:")
    print(df.info())
    print(":::::::")
    print("Shape:")
    print(df.shape)
    print(":::::::")
    print("Describe:")
    print(df.describe())

def train_NE_Reg(df, split, Target):
    target = Target
    Input = df

    X_train, X_test, y_train, y_test = model_selection.train_test_split(Input.values, target.values, test_size=split, random_state=42)
    NE_reg = linear_model.LogisticRegression(solver='liblinear').fit(X_train, y_train)
    print("R2 Score Logistic Regression: ")
    print(NE_reg.score(X_test,y_test))  #Uses R2 score for testing
    # print("MSE: ")
    # y_pred = NE_reg.predict(X_test)
    # print(mean_squared_error(y_test, y_pred))
    return NE_reg.score(X_test,y_test)

def train_SGD_Log_Reg(df,split,target):
    Input = df
    Input1 = Input.values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(Input1, target.values, test_size=split, random_state=42)
    SGD_reg2 = SGDClassifier(loss = 'log', alpha=0,learning_rate='constant', eta0=0.01, random_state=28, penalty='elasticnet').fit(X_train,y_train)
    print("R2 score SGD: ")
    print(SGD_reg2.score(X_test,y_test))   #almost as good as NE

def train_MLP(df, split, target):
    Input = df
    X_train, X_test, y_train, y_test = model_selection.train_test_split(Input.values, target.values, test_size=split, random_state=42)
    NE_reg = MLPClassifier(random_state=42, max_iter=500).fit(X_train, y_train)
    print("R2 Score MLP: ")
    print(NE_reg.score(X_test,y_test))  #Uses R2 score for testing
    return NE_reg.score(X_test,y_test)

def cv_NE_Reg(X,y):
    scores = []

    kf = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        NE_reg = linear_model.LogisticRegression(solver='liblinear').fit(X_train, y_train)
        scores.append(NE_reg.score(X_test,y_test))

    print('CV NE_Reg avg:')
    print(np.mean(scores))

def cv_SGD(X,y):
    scores = []

    kf = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        SGD_reg2 = SGDClassifier(loss = 'log', alpha=0,learning_rate='constant', eta0=0.01, random_state=28, penalty='l1').fit(X_train,y_train)
        scores.append(SGD_reg2.score(X_test,y_test))

    print('SGD avg:')
    print(np.mean(scores))

def cv_MLP(X,y):
    scores = []

    kf = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        MLP = MLPClassifier(random_state=42, max_iter=600).fit(X_train, y_train)
        scores.append(MLP.score(X_test,y_test))

    print('CV MLP avg:')
    print(np.mean(scores))

def knn_cross_val(X, y):
    for i in range(16):
        knn = KNeighborsClassifier(n_neighbors=(i+1))
        # knn.fit(X_train, y_train)
        # print('KNN score:', knn.score(X_test, y_test))  #Without cv
        score = cross_val_score(knn, X, y, cv=5)
        print('KNN cross val', i, ':')
        print(np.mean(score))

def Find_CV_ALL(X,y):
    random_forest = RandomForestClassifier(bootstrap=True, criterion='gini', max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=20)
    # random_forest.fit(X_train, y_train)
    score = cross_val_score(random_forest, X, y, cv=5)
    print("Random Forest CV:")
    print(np.mean(score))
    

    MLP = MLPClassifier(max_iter=600)
    score = cross_val_score(MLP, X, y, cv=5)
    print("MLP CV:")
    print(np.mean(score))

    SGD_reg2 = SGDClassifier(loss = 'log', alpha=0.01, learning_rate='constant', eta0=0.1, penalty='l2', max_iter=5000)
    Input1 = StandardScaler().fit_transform(X)
    score = cross_val_score(SGD_reg2, Input1, y, cv=5)
    print("SGD CV:")
    print(np.mean(score))
    # {'alpha': 0.01, 'eta0': 0.01, 'learning_rate': 'constant', 'loss': 'modified_huber', 'max_iter': 10000, 'penalty': 'elasticnet'}
    # {'alpha': 0.01, 'eta0': 0.1, 'learning_rate': 'constant', 'loss': 'log', 'max_iter': 5000, 'penalty': 'l2'}

    NE_reg = linear_model.LogisticRegression(solver='liblinear')
    score = cross_val_score(NE_reg, X, y, cv=5)
    print("Logistic Regression CV:")
    print(np.mean(score))

def Tune(X,y,model):
    # params = {"bootstrap":["True","False"],"max_features" : ["auto", "sqrt", "log2"],"n_estimators": [10,20,30,100,200,300],'criterion': ["gini", "entropy"], "min_samples_leaf" : [1,3,5,7,9,15,20],"min_samples_split": [2,4,6,8,10,12,18,20,40]}
    # params = {'loss':['hinge','log','modified_huber','squared_hinge','perceptron','squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive','log'], 'penalty':['l1','l2','elasticnet'],
    # 'alpha':[0.00001,0.0001, 0.001, 0.01, 0.1], 'learning_rate':['constant','optimal','invscaling','adaptive'], 'eta0':[0.001, 0.01, 0.1], 'max_iter':[1000,5000,10000,50000,100000]}
    #"penalty":['l1','l2','elasticnet','none'], 'dual':["True", "False"], 
    params = {'activation':['identity','logistic','tanh','relu'], 'solver':['sgd','adam'], 'alpha':[0.00001,0.0001, 0.001, 0.01, 0.1], 
    'learning_rate':['constant','invscaling','adaptive'], 'max_iter':[300,600,1000], 'hidden_layer_sizes':[(100,),(10,30,10)]}
    # params = {'base_estimator':[linear_model.LogisticRegression(solver='liblinear'), SGDClassifier(loss = 'log', alpha=0.01, learning_rate='constant', eta0=0.1, penalty='l2', max_iter=5000)],
    # 'n_estimators':[10,20,50,100,150,200], 'max_features':[1.0,1.5,2.0,2.5], 'max_samples':[1.0,1.5,2.0,2.5], 'bootstrap':['True','False']}
    # X = np.array(X,dtype=np.float64)
    gs = model_selection.GridSearchCV(model, param_grid=params, cv=5)
    # Input1 = StandardScaler().fit_transform(X)
    gs.fit(X,y)
    print(gs.best_params_)

df = pd.read_csv("train.csv")
test = pd.read_csv('test.csv')
# info_on_data(df)

print(df.isnull().sum())

# df.drop('Cabin',inplace=True, axis=1)
# test.drop('Cabin',inplace=True, axis=1)

df['Cabin'] = df['Cabin'].str[:1]
test['Cabin'] = test['Cabin'].str[:1]

cabin_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
df['Cabin'] = df['Cabin'].map(cabin_map)
test['Cabin'] = test['Cabin'].map(cabin_map)

df["Cabin"].fillna(df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 4, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

df['Title'] = df['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)


df.drop('Name', inplace=True, axis=1)
df.drop('PassengerId', inplace=True, axis=1)
df.drop('Ticket', inplace=True, axis=1)

test.drop('Name', inplace=True, axis=1)
pass_id = test['PassengerId'].values
test.drop('PassengerId', inplace=True, axis=1)
test.drop('Ticket', inplace=True, axis=1)

m = df['Age'].mean()

# df['Age'].fillna(m,inplace=True)
df["Age"].fillna(df.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

df['Embarked'] = df['Embarked'].fillna('S')
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

# df.dropna(inplace=True)

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
df['Embarked'] = df['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)


''' ONE HOT ENCONDING

# df = pd.get_dummies(df)

# df['Embarked_C'] = df['Embarked_C'].astype(int)
# df['Embarked_Q'] = df['Embarked_Q'].astype(int)
# df['Embarked_S'] = df['Embarked_S'].astype(int)
# df['Sex_male'] = df['Sex_male'].astype(int)

# df.drop('Sex_female', inplace=True, axis=1)

'''

gender_map = {'female':1, 'male':0}
df['Sex'] = df['Sex'].map(gender_map)  
test['Sex'] = test['Sex'].map(gender_map)  

info_on_data(df)

df['ageXclass'] = df['Age']*df['Pclass']
test['ageXclass'] = test['Age']*test['Pclass']

df['sexYclass'] = df['Sex']*df['Pclass']
test['sexYclass'] = test['Sex']*test['Pclass']

df['Fare/Person'] = df['Fare']/(df['Parch']+df['SibSp']+1)
test['Fare/Person'] = test['Fare']/(test['Parch']+test['SibSp']+1)

df['nr_fam'] = df['Parch']+df['SibSp']
test['nr_fam'] = test['Parch']+test['SibSp']

df.drop('Parch',axis=1,inplace=True)
test.drop('Parch',axis=1,inplace=True)

df.drop('Embarked',axis=1,inplace=True)
test.drop('Embarked',axis=1,inplace=True)

df.drop('SibSp',axis=1,inplace=True)
test.drop('SibSp',axis=1,inplace=True)

# df.drop('Pclass',axis=1,inplace=True)
# test.drop('Pclass',axis=1,inplace=True)

# df_male = df[(df['Sex_male'] == 1)]
# df_male['Age'].plot(kind='box')
#df_male.plot(kind='scatter', x='Age', y='Survived', title='Revenue (millions) vs Rating')
# sns.pairplot(df, hue='Survived')
# plt.show()

target = df['Survived']
df.drop('Survived', inplace=True, axis=1)
X = df.values
y = target.values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


'''Base Values

df_sub = df[['Pclass', 'Age', 'SibSp', 'Fare', 'Parch', 'Sex_male']]

train_NE_Reg(df_sub, 0.25, target)
train_SGD_Log_Reg(df, split=0.25, target=target)
train_MLP(df, 0.25, target)

'''

'''CV BASE VALS

cv_NE_Reg(X,y)
cv_MLP(X,y)
cv_SGD(X,y)

'''

''' Feature Selections
print(df.head())
rfe = RFECV(estimator=linear_model.LogisticRegression(solver='liblinear'), step=1, cv=5)
rfe = rfe.fit(X, y)
print(rfe.support_)
'''

''' Lasso 
model = linear_model.Lasso(alpha=0.1)
model.fit(X_train, y_train)
print(model.intercept_)
print(model.predict(X_test))
print(model.score(X_test, y_test))
'''

bag = BaggingClassifier(base_estimator=linear_model.LogisticRegression(solver='liblinear'), bootstrap=True, n_estimators=50)
# {'base_estimator': LogisticRegression(solver='liblinear'), 'bootstrap': 'True', 'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 50}
score = cross_val_score(bag, X, y, cv=5)
# bag.fit(X_train, y_train)
print('With Bagging:')
# print(bag.score(X_test, y_test))
print((score))
bag.fit(X,y)

boost = AdaBoostClassifier(algorithm='SAMME',base_estimator=linear_model.LogisticRegression(solver='liblinear'),n_estimators=50,learning_rate=1.5)
# {'algorithm': 'SAMME', 'base_estimator': LogisticRegression(solver='liblinear'), 'learning_rate': 1.5, 'n_estimators': 50}
score = cross_val_score(boost, X, y, cv=5)
# boost.fit(X_train, y_train)
print('With Boosting')
# print(boost.score(X_test, y_test))
print((score))
boost.fit(X,y)

random_forest = RandomForestClassifier(bootstrap=True, criterion='gini', max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=100)
# random_forest.fit(X_train, y_train)
score = cross_val_score(random_forest, X, y, cv=5)
print('Random Forest:')
# print(random_forest.score(X_test, y_test))
print((score))

random_forest.fit(X,y)

# ''' Feature Importance Random Forest
ft_imp = pd.Series(random_forest.feature_importances_, index=df.columns).sort_values(ascending=False)
ft_imp.plot.bar()
plt.show()
# print(ft_imp.head(10))
# '''

'''  Random Forest Tuning
params = {'max_depth':[1,5,10,20,30],"bootstrap":["True","False"],"max_features" : ["auto", "sqrt", "log2"],"n_estimators": [10,20,30,100,200,300],'criterion': ["gini", "entropy"], "min_samples_leaf" : [1,3,5,7,9,15,20],"min_samples_split": [2,4,6,8,10,12,18,20,40]}
# params = {"max_features" : ["auto", "sqrt", "log2"],"n_estimators": [100,200,300,500],'criterion': ["gini", "entropy"], "min_samples_leaf" : [1,3,5], "min_samples_split": [2,4,6,8,10,15]}
rf = RandomForestClassifier()
gs = model_selection.GridSearchCV(rf, param_grid=params, cv=5)
gs.fit(X,y)
print(gs.best_params_)
# RESULT: {'bootstrap': 'True', 'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 20}
# RESULT: {'bootstrap': 'False', 'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_leaf': 5, 'min_samples_split': 8, 'n_estimators': 10}
# RESULT: {'bootstrap': 'False', 'criterion': 'entropy', 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 10}
# RESULT: {'bootstrap': 'True', 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 20}
'''



Find_CV_ALL(X,y)

''' MLP 
MLP = MLPClassifier(activation='tanh', alpha=0.01, hidden_layer_sizes=(10,30,10), learning_rate='adaptive', max_iter=5000, solver='adam')
score = cross_val_score(MLP, X, y, cv=5)
print('MLP:')
# print(random_forest.score(X_test, y_test))
print((score))
MLP.fit(X,y)
'''
# model = MLPClassifier()
# Tune(X,y,model)

# MLP: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (10, 30, 10), 'learning_rate': 'adaptive', 'max_iter': 5000, 'solver': 'adam'}

# svm = SVC()
# score = cross_val_score(svm, X, y, cv=5)
# print('SVC:')
# print(np.mean(score))

model = linear_model.LogisticRegression(solver='liblinear')
model.fit(X,y)

print(";;;")

### Sub Stuff

info_on_data(test)
test_set = test.values

predictions = random_forest.predict(test_set)

test_dict = {'PassengerId':pass_id, 'Survived':predictions}


test_df = pd.DataFrame(test_dict)
test_df.to_csv('Preds.csv', index=False)