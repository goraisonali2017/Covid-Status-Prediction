# importing required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns



# Reading the csv file
df_covid = pd.read_csv('covidcare.csv')


# gathering some information about the output variable
sns.set_style('whitegrid')
sns.countplot(x='covid_status',data=df_covid,palette='RdBu_r')
df_covid.covid_status.value_counts()



# lets take first 25000 data and check if the ratio is acceptable
df_covid = df_covid[:25000]
sns.set_style('whitegrid')
sns.countplot(x='covid_status',data=df_covid,palette='RdBu_r')
df_covid.covid_status.value_counts()




# gathering information about the null values in dataframe
sns.heatmap(df_covid.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# knowing more about data using info() method
print(df_covid.info())
# finding which features really matter in predicting the target variable
print(df_covid.corr())


# visualizing the corelation so that we can get to know about the relation between
# the features at a glance
plt.figure(figsize=(10,5))
sns.heatmap(df_covid.corr(),annot=True,annot_kws={"size":15})



print(type(df_covid.hypertension[0]),type(df_covid.asthma[0]),type(df_covid.heart_diasease[0]))


df_covid.hypertension.replace(["yes","no"],[1,0],inplace=True)
df_covid.asthma.replace(["yes","no"],[1,0],inplace=True)
df_covid.heart_diasease.replace(["yes","no"],[1,0],inplace=True)
print(type(df_covid.hypertension[0]),type(df_covid.asthma[0]),type(df_covid.heart_diasease[0]))



# checking the dependence of hypertension and asthma
sns.set_style('whitegrid')
sns.countplot(x='hypertension',hue='asthma',data=df_covid,palette='RdBu_r')



# checking the dependence of asthma and heart-disease
sns.set_style('whitegrid')
sns.countplot(x='asthma',hue='heart_diasease',data=df_covid,palette='RdBu_r')



# checking the dependence of heart-disease and hypertension
sns.set_style('whitegrid')
sns.countplot(x='heart_diasease',hue='hypertension',data=df_covid,palette='RdBu_r')



print ("0 - Male =", len(df_covid[(df_covid.covid_status == 0) & (df_covid.gender == 'M')]), end= ", ")
print ("1 - Male =", len(df_covid[(df_covid.covid_status == 1) & (df_covid.gender == 'M')]), end= ", ")




print ("0 - Female =", len(df_covid[(df_covid.covid_status == 0) & (df_covid.gender == 'F')]), end= ", ")
print ("1 - Female =", len(df_covid[(df_covid.covid_status == 1) & (df_covid.gender == 'F')]), end= ", ")



#replacing "M" and "F" in gender column as 1,0
df_covid.gender.replace(["M","F"],[1,0],inplace=True)



sns.set_style('whitegrid')
sns.countplot(x='covid_status',hue='blood_group',data=df_covid,palette='rainbow')



# replacing the blood groups with their rate of availability
df_covid.blood_group.replace(["AB-","B-","A-","O-","A+","AB+","B+","O+"],[0,1,2,3,4,5,6,7],inplace=True)



sns.set_style('whitegrid')
sns.countplot(x='covid_status',hue='smoking_status',data=df_covid,palette='rainbow')



#replacing "smokes","formerly smoked","never smoked" in smoking_status column as 2,1,0
df_covid.smoking_status.replace(["smokes","formerly smoked","never smoked"],[2,1,0],inplace=True)



sns.set_style('whitegrid')
sns.countplot(x='covid_status',hue='profession',data=df_covid,palette='rainbow')




# calculating the median of age as it will be needed to replace the null values of bmi
med_age = np.median(df_covid[['age']].dropna())
print ("Median of age for covid_status-1 =", med_age, end= ", ")



# function that will help us to replace the null values of age with its ,median values
def impute_age(cols):
    age = cols[0]
    if pd.isnull(age):
        return med_age
    else:
        return age




# replacing the null values of age with its median value
df_covid['age']=df_covid[['age']].apply(impute_age,axis=1)



# calculating the median of avg_glucose as it will be needed to replace the null values of bmi
med_avg_glucose = np.median(df_covid[['avg_glucose']].dropna())
print ("Median of avg_glucose for covid_status-1 =", med_avg_glucose, end= ", ")


# function that will help us to replace the null values of age with its ,median values
def impute_avg_glucose(cols):
    avg_glucose = cols[0]
    if pd.isnull(avg_glucose):
        return med_avg_glucose
    else:
        return avg_glucose




# replacing the null values of avg_glucose with its median value
df_covid['avg_glucose']=df_covid[['avg_glucose']].apply(impute_avg_glucose,axis=1)



# calculating the median of bmi as it will be needed to replace the null values of bmi
med_bmi = np.median(df_covid[['bmi']].dropna())
print ("Median of bmi for strokes-1 =", np.median(df_covid[['bmi']].dropna()), end= ", ")



# function that will help us to replace the null values of bmi with its ,median values
def impute_bmi(cols):
    bmi = cols[0]
    if pd.isnull(bmi):
        return med_bmi
    else:
        return bmi



# replacing the null values of bmi with its median value 
df_covid['bmi']=df_covid[['bmi']].apply(impute_bmi,axis=1)



# calculating the median of body_temp as it will be needed to replace the null values of bmi
med_body_temp = np.median(df_covid[['body_temp']].dropna())
print ("Median of body_temp for covid_status-1 =", med_body_temp, end= ", ")


# function that will help us to replace the null values of body_temp with its ,median values
def impute_body_temp(cols):
    body_temp = cols[0]
    if pd.isnull(body_temp):
        return med_body_temp
    else:
        return body_temp



# replacing the null values of body_temp with its median value 
df_covid['body_temp']=df_covid[['body_temp']].apply(impute_body_temp,axis=1)




#dropping profession column as it has got no direct relation with the covid_status
df_covid.drop(["profession"], inplace = True, axis = 1)



#dropping id column as it has got no direct relation with the covid_status
df_covid.drop(["id"], inplace = True, axis = 1)



#dropping name column as it has got no direct relation with the covid_status
df_covid.drop(["name"], inplace = True, axis = 1)



#dropping email column as it has got no direct relation with the covid_status
df_covid.drop(["email"], inplace = True, axis = 1)



# cross-checking if the values are correctly replaced or not
sns.heatmap(df_covid.isnull(),yticklabels=False,cbar=False,cmap='viridis')



# cross-checking the data as well using info method
print(df_covid.info())


#Lets check whether the data is ready for implementing our Machine Learning Models or not
df_covid


#Importing the necessary modules required for building our Models
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support



#Logistic Regression

from sklearn.linear_model import LogisticRegression

# splitting the data for training and testing
# 70% of the data are kept for training and 30% of the data are kept for the testing
x_train_logistics,x_test_logistics,y_train_logistics,y_test_logistics=train_test_split(df_covid.drop('covid_status',axis=1),df_covid['covid_status'],test_size=0.30,random_state=101)

# validating if the rows are correctly divided
print("No. of Train rows -> ",len(y_train_logistics), 0.70 * df_covid.shape[0])
print("No. of Test rows -> ",len(y_test_logistics), 0.30 * df_covid.shape[0])
print(x_train_logistics.shape)
print(y_train_logistics.shape)


# printing if the train and test data are correctly seperated
print(x_train_logistics.head())
print(y_train_logistics.head())
print(x_test_logistics.head())
print(y_test_logistics.head())



# creating an object of LogisticRegression() class which will help us in 
# training and testing our model
logmodel=LogisticRegression()

# fitting the training data so that model learns from previous data
logmodel.fit(x_train_logistics,y_train_logistics)




# predicting the output of the model for the testing data
# data testing is very important as based on it we will decide if the model is 
# correctly trained and accurate or not
prediction_logistics=logmodel.predict(x_test_logistics)
# printing if model successfully predicted or not
print(prediction_logistics,len(prediction_logistics))



# print(classification_report(y_test,predictions))
# checking the accuracy of the model
cr_logistics = classification_report(y_test_logistics, prediction_logistics)
print(cr_logistics)



# checking the correct and incorrect predicted outputs w.r.t. actual data
cm_logistic = confusion_matrix(y_test_logistics, prediction_logistics)
print(cm_logistic)
confusion_df = pd.DataFrame(confusion_matrix(y_test_logistics,prediction_logistics), 
             columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
             index = ["Class " + str(class_name) for class_name in [0,1]])

print(confusion_df)



prfs_logistics = precision_recall_fscore_support(y_test_logistics, prediction_logistics)
prfs_logistics


#printing the coefficients
print(logmodel.coef_)



# printing the intercept
print(logmodel.intercept_)



import joblib
joblib.dump(logmodel ,"covid_predict.pkl")


Final_Model =joblib.load("covid_predict.pkl")



Final_Model.predict(x_test_logistics)






