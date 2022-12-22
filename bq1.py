from sklearn.naive_bayes import GaussianNB # Assigning features and label variables
from sklearn import preprocessing
weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast','Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',  'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes','No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
le = preprocessing.LabelEncoder() # creating labelEncoder
wheather_encoded = le.fit_transform(weather) # Converting string labels into numbers.
print(wheather_encoded)
temp_encoded = le.fit_transform(temp)# Converting string labels into numbers
label = le.fit_transform(play)
print("Temp:", temp_encoded)
print("Play:", label)
features = zip(wheather_encoded, temp_encoded)# Combinig weather and temp into single listof tuples
print(features)
model = GaussianNB()# Create a Gaussian Classifier
model.fit(features, label)# Train the model using the training setspredicted = model.predict([[0, 2]]) # Predict Output   # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)
#---------------------------------------------------------------------------------------
# 1) Consider following dataset 
# weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','S
# unny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'] 
# temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mi
# ld','Mild','Hot','Mild'] 
# play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Y
# es','No']. Use Naïve Bayes algorithm to predict[ 0:Overcast, 2:Mild] 
# tuple belongs to which class whether to play the sports or not. 
