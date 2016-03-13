from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import matplotlib
import numpy as np
import csv
import copy
from sklearn import preprocessing
from patsy import dmatrices, dmatrix
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from operator import itemgetter
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.grid_search import RandomizedSearchCV
#Read data and configuration parameters#

test_file= 'c:/users/jinji/desktop/titanic/test.csv'
submission_path = 'c:/users/jinji/desktop/titanic/submision.csv'
model_file = 'c:/users/jinji/desktop/test/model-rf.pkl'
seed = 1

def clean_cabin(x):
    try:
        x[0]
        return 'Cabin'
    except TypeError:
        return 'NoCabin'
def titleSearch(full_name, titleList):
    for title in titleList:
        if full_name.find(title) != -1:
            return title
    return np.nan
def replaceTitle(x):
    title = x.Title
    if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Counteness','Mme','Mrs']:
        return 'Mrs'
    elif title in ['Mlle','Ms','Miss']:
        return 'Miss'
    elif title =='Dr':
        if x.Sex=='male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title == '':
        if x.Sex == 'male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title

le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()

def feature_engineering(df):
    # process fare
    df.Fare = df.Fare.map(lambda x:np.nan if x==0 else x)
    # preprocess titles
    title_list = ['Mrs','Mr','Master','Miss','Major','Rev','Dr','Ms','Mlle','Col','Capt','Mme','Countess','Don','Jonkheer']
    df['Title'] = df.Name.map(lambda x: titleSearch(x,title_list))
    df.Title = df.apply(replaceTitle,axis=1)
    # create new family_size
    df['Family_size'] = df.SibSp + df.Parch +1
    df['Family'] = df.SibSp * df.Parch
    # input nan values
    df.loc[(df.Fare.isnull()) & (df.Pclass==1),'Fare'] = np.median(df[df['Pclass']==1]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass==2),'Fare'] = np.median(df[df['Pclass']==2]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass==3),'Fare'] = np.median(df[df['Pclass']==3]['Fare'].dropna())
    df.Embarked.fillna('S',inplace=True)

    df['AgeFill'] = df.Age
    mean_ages = np.zeros(4)
    mean_ages[0]=np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1]=np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2]=np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3]=np.average(df[df['Title'] == 'Master']['Age'].dropna())
    df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'AgeFill'] = mean_ages[3]

    df['AgeCat'] = df.AgeFill
    df.loc[ (df.AgeFill<=10) ,'AgeCat'] = 'child'
    df.loc[ (df.AgeFill>60),'AgeCat'] = 'aged'
    df.loc[ (df.AgeFill>10) & (df.AgeFill <=30) ,'AgeCat'] = 'adult'
    df.loc[ (df.AgeFill>30) & (df.AgeFill <=60) ,'AgeCat'] = 'senior'

    df['Gender'] = df.Sex.map({'female':0, 'male':1}).astype(int)
    #Cabin info process
    df.Cabin = df.Cabin.apply(clean_cabin)
    dummy = pd.get_dummies(df.Cabin)
    df.drop(['Cabin'],axis=1, inplace=True)
    df = df.join(dummy.Cabin)
    #fare per person
    df['Fare_per_person'] = df.Fare/df.Family_size
    #age times class
    df['AgeClass'] = df.AgeFill * df.Pclass
    df['ClassFare'] = df.Pclass * df.Fare_per_person

    df['HighLow'] = df.Pclass
    df.loc[(df.Fare_per_person<8),'HighLow'] = 'Low'
    df.loc[(df.Fare_per_person>=8),'HighLow'] = 'High'

    le.fit(df['Sex'])
    x_sex=le.transform(df['Sex'])
    df['Sex']=x_sex.astype(np.float)

    le.fit(df['HighLow'])
    x_hl = le.transform(df['HighLow'])
    df['HighLow'] = x_hl.astype(np.float)

    le.fit( df['Ticket'])
    x_Ticket=le.transform( df['Ticket'])
    df['Ticket']=x_Ticket.astype(np.float)

    le.fit(df['Title'])
    x_title=le.transform(df['Title'])
    df['Title'] =x_title.astype(np.float)

    le.fit(df['AgeCat'])
    x_age=le.transform(df['AgeCat'])
    df['AgeCat'] =x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb=le.transform(df['Embarked'])
    df['Embarked']=x_emb.astype(np.float)

    df = df.drop(['PassengerId','Name','Age'], axis=1)

    return df

#read data

testdf=pd.read_csv(test_file)

ID=testdf['PassengerId']
##clean data
df_test=feature_engineering(testdf)
#df_test['Survived'] =  [0 for x in range(len(df_test))]

df_test = df_test.drop(['SibSp','Parch','Ticket','Embarked','Family','AgeFill','Cabin'],axis=1)

print df_test.info()

clf = joblib.load(model_file)
test_predict=clf.predict(df_test).astype(int)
dfjo = DataFrame(dict(PassengerId=ID,Survived=test_predict), columns=['PassengerId','Survived'])
dfjo.to_csv(submission_path,index_label=None,index_col=False,index=False)
'''
formula_ml='Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 

y_p,x_test = dmatrices(formula_ml, data=df_test, return_type='dataframe')
y_p = np.asarray(y_p).ravel()
print  y_p.shape,x_test.shape
#serialize training
model_file=MODEL_PATH+'model-rf.pkl'
clf = joblib.load(model_file)
####estimate prediction on test data set
y_p=clf.predict(x_test).astype(int)
print y_p.shape

outfile=SUBMISSION_PATH+'prediction-BS.csv'
dfjo = DataFrame(dict(Survived=y_p,PassengerId=ID), columns=['Survived','PassengerId'])
dfjo.to_csv(outfile,index_label=None,index_col=False,index=False)
'''