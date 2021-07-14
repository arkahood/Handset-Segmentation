import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model
from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
from sklearn import feature_selection
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier

handset = pd.read_csv("C:\\Users\\ARKA\\Desktop\\Machine Learning Project\\handset_segmentation\\train.csv")

# =============================================================================
# outlier removal :
# =============================================================================
handset1 = handset.drop(handset[(handset["fc"]>16) | (handset["px_height"]>1944)].index)
handset1.shape
# =============================================================================
# without standardize data :
# =============================================================================
y = handset1["price_range"]
X = handset1.drop("price_range",axis=1)
Xtrain,Xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=.2,random_state=0)
X_conti = Xtrain.drop(["blue","dual_sim","three_g","four_g","touch_screen","wifi"],axis=1)
X_discrete = Xtrain[["blue","dual_sim","three_g","four_g","touch_screen","wifi"]]
Xtest_conti = Xtest.drop(["blue","dual_sim","three_g","four_g","touch_screen","wifi"],axis=1)
Xtest_discrete = Xtest[["blue","dual_sim","three_g","four_g","touch_screen","wifi"]]
# =============================================================================
# Standardize data :
# =============================================================================
scaler = preprocessing.StandardScaler()
scaler.fit(X_conti)
X_conti_scl=scaler.transform(X_conti)
Xtest_scl = scaler.transform(Xtest_conti)
X_scl=pd.DataFrame(X_conti_scl)
X_scl.columns=X_conti.columns
X_scl.index = X_conti.index
Xtest_scl=pd.DataFrame(Xtest_scl)
Xtest_scl.columns=X_conti.columns
Xtest_scl.index = Xtest_conti.index


Xtrain_final = pd.concat([X_scl,X_discrete],axis=1)
Xtest_final = pd.concat([Xtest_scl,Xtest_discrete],axis=1)

# =============================================================================
# mutual info
# =============================================================================
miscore=feature_selection.mutual_info_classif(X_conti_scl,ytrain)
miser = pd.Series(miscore)
miser.index = X_conti.columns
miser.sort_values(ascending = False).plot.bar()
miser.sort_values(ascending = False)[:5]

# =============================================================================
# anova 
# =============================================================================
fvalue,pvalue = feature_selection.f_classif(X_conti_scl,ytrain)
for colname,fv,pv in zip(Xtrain.columns,fvalue,pvalue):     #lowest pvalue will be the highest influence on y
    print(colname,fv,pv)
qwe=pd.Series(pvalue)
qwe.index = X_conti.columns
qwe.sort_values(ascending=True).plot.bar()
qwe.sort_values(ascending=True)[:5]

# =============================================================================
# chi square :
# =============================================================================
cval,pval = obj = feature_selection.chi2(X_discrete,ytrain)
ser = pd.Series(pval)
ser.index = X_discrete.columns
ser.sort_values(ascending=True).plot.bar()
ser.sort_values(ascending=True)[:]

# =============================================================================
# Manually selecting data :
# =============================================================================
Xtrain_manul = Xtrain[["ram","battery_power","px_height","mobile_wt","int_memory","touch_screen","four_g","sc_h","n_cores"]]
Xtest_manul = Xtest[["ram","battery_power","px_height","mobile_wt","int_memory","touch_screen","four_g","sc_h","n_cores"]]
# =============================================================================
# logistic regression :
# =============================================================================
obj = feature_selection.SelectFromModel(estimator=linear_model.LogisticRegression(),max_features=8,threshold=-np.inf)
obj.fit(Xtrain_final,ytrain)
obj.get_support()
Xtrain.columns[obj.get_support()]
Xtrain_std1 = obj.transform(Xtrain_final)
model = linear_model.LogisticRegression(penalty="l2",C=10.0)
model.fit(Xtrain_std1,ytrain)
predtrain=model.predict(Xtrain_std1)
Xtes = obj.transform(Xtest_final)
predtest = model.predict(Xtes)
metrics.f1_score(ytrain,predtrain,average="macro")
metrics.f1_score(ytest,predtest,average="macro")
metrics.confusion_matrix(ytest,predtest)

# =============================================================================
# KNN :
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

# creating odd list of K for KNN
neighbors = list(range(1, 50, 2))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')
    scores = model_selection.cross_val_score(knn, Xtrain, ytrain, cv=10, scoring=scorer)
    cv_scores.append(scores.mean())



# changing to misclassification error
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal number of neighbors is {}".format(optimal_k))

# plot misclassification error vs k
plt.plot(neighbors, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Misclassification Error")
plt.show()

knn1 = KNeighborsClassifier(n_neighbors=optimal_k)
knn1.fit(Xtrain,ytrain)
predtrain1=knn1.predict(Xtrain)
predtest1 = knn1.predict(Xtest)
metrics.f1_score(ytrain,predtrain1,average="macro")
metrics.recall_score(ytrain,predtrain1,average="macro")
metrics.precision_score(ytrain,predtrain1,average="macro")
metrics.confusion_matrix(ytrain,predtrain1)

metrics.f1_score(ytest,predtest1,average="macro")
metrics.recall_score(ytest,predtest1,average="macro")
metrics.precision_score(ytest,predtest1,average="macro")
metrics.accuracy_score(ytest,predtest1)
metrics.confusion_matrix(ytrain,predtrain1)           
  
# =============================================================================
# Decision tree:
# =============================================================================

hsdict = {"max_depth":list(range(4,21)),"min_samples_split":list(range(4,21)),"max_features":list(range(2,15))}
scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')
gridobj = model_selection.GridSearchCV(estimator=tree.DecisionTreeClassifier(random_state=0),param_grid=hsdict,scoring=scorer)
gridobj.fit(Xtrain,ytrain)
print(gridobj.best_params_)

model1 = tree.DecisionTreeClassifier(criterion="gini",random_state = 0, max_depth=7, min_samples_split=15,max_features=12)
model1.fit(Xtrain,ytrain)
predtrain = model1.predict(Xtrain)
predtest = model1.predict(Xtest)
metrics.f1_score(ytrain,predtrain,average="macro")
metrics.f1_score(ytest,predtest,average="macro")

# =============================================================================
# random forest :
# =============================================================================
scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')
gridobj = model_selection.GridSearchCV(estimator=ensemble.RandomForestClassifier(random_state=0),param_grid=hsdict,scoring=scorer)
#{'max_depth': 13, 'max_features': 11, 'min_samples_split': 6}
obj = feature_selection.SelectFromModel(ensemble.RandomForestClassifier(),max_features=11)
obj.fit(Xtrain,ytrain)
Xtrain_ranfor = obj.transform(Xtrain)
Xtest_ranfor = obj.transform(Xtest)
model = ensemble.RandomForestClassifier(n_estimators=10,random_state=0)
model.fit(Xtrain_ranfor,ytrain)
metrics.f1_score(ytrain,predtrain,average="macro")
metrics.f1_score(ytest,predtest,average="macro")

# =============================================================================
# Naive bayes :
# =============================================================================
from sklearn.naive_bayes import GaussianNB


obj=feature_selection.RFE(estimator=naive_bayes.GaussianNB(),n_features_to_select=20)
obj.fit(Xtrain,ytrain)
obj.transform(Xtrain)    
predtest=obj.predict(Xtest)
print(metrics.recall_score(ytest,predtest,average="macro"))




