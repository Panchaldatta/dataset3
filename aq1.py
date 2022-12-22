
from ast import Yield
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
pima = pd.read_csv("C:/Users/hp/Downloads/diabetes.csv")
pima.head()
import seaborn as sns
corr=pima.corr()
ax=sns.heatmap(corr,vmin=1,vmax=1,center=0,cmap=sns.diverging_palette(20,220,n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
feature_cols=['Pregancies','Insulin','BMI','Age','Gluocse','BloodPressure','DiabetesPedigreeFunction']#feature selection
x=pima.Outcome
X_train,X_test,Y_train=train_test_split(x,Yield,test_size=0.3,random_state=1)#split data
classifier=DecisonTreeClassifier()#build model
classifier=classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)#predict
print(Y_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,y_pred)
print(confusion_matrix(Y_test,y_pred))
print("Accuracy:",metric.accuracy_score(Y_test,y_pred))#accuracys
from six import StringIO
from IPython.display import image
from sklearn.tree import export_graphviz
import pydotplus
dot_data=StringIO()
export_graphviz(classifier,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=feature_cols,class_names=['0','1'])
graph=pydotsplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())


# 1) Write a Python program build Decision Tree Classifier using Scikit-learn 
# package for diabetes data set (download database from 
#https://www.kaggle.com/uciml/pima-indians-diabetes-database) 


