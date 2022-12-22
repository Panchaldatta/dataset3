from sklearn import datasets
cancer=datasets.load_breast_cancer()
print("Features:",cancer.feature_names)
print("Labels:",cancer.target_names)
cancer.data.shape
print(cancer.data[0:5])
print(cancer.target)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,
test_size=0.3,random_state=109)#70% training and 30% test
from sklearn import svm
clf=svm.SVC(kernel='linear')#Linear kernel
clf.fit(X_train.y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
#--------------------------------------------------------------------------------------------




# 1) Write a Python program to build SVM model to Cancer dataset. The 
# dataset is available in the scikit-learn library. Check the accuracy of model 
# with precision and recall.