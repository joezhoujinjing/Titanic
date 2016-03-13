from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
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

def gender(x):
	print x.Sex
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


#Configurations
train = pd.read_csv('c:/users/jinji/desktop/titanic/train.csv')
test= pd.read_csv('c:/users/jinji/desktop/titanic/test.csv')
submission_path = 'c:/users/jinji/desktop/titanic/submision.csv'
seed = 3

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

	#dead_age = df.Age[df.Survived==0]
	#dead_fare = df.Fare[df.Survived==0]
	#alive_age = df.Age[df.Survived==1]
	#alive_fare = df.Fare[df.Survived==1]

	df['Gender'] = df.Sex.map({'female':0, 'male':1}).astype(int)
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


	#plt.scatter(dead_age,dead_fare,color = 'red')
	#plt.scatter(alive_age,alive_fare,color = 'green')

	#plt.scatter(df.PassengerId[df.Survived==0],df.Age[df.Survived==0],color='red')
	#plt.scatter(df.PassengerId[df.Survived==1],df.Age[df.Survived==1],color='green')
	#print 'death averge fare price',dead_fare.mean()
	#print 'survival averge fare price',alive_fare.mean()

	#Gender analysis
	#female = df[df.Sex == 'female'][df.Age>16]
	#male = df[df.Sex == 'male'][df.Age>16]
	#child = df[df.Age<=16]
	#print 'female survived',female.Survived.mean()
	#print 'male survived',male.Survived.mean()
	#print 'child survived',child.Survived.mean()
	#Cabin analysis
	#df.Cabin.fillna('',inplace=True)
	#print 'NoCabin survival rate',df[df.Cabin=='Missing'].Survived.mean()
	#print 'Cabin survival rate',df[df.Cabin!='Missing'].Survived.mean()
	#for c in 'ABCDEFG':
	#	print 'Cabin %c survival rate'%c,df[df.Cabin==c].Survived.mean()
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

	#pclass analysis
	#print 'class survival rate', df[df.Pclass==1].Survived.mean()
	#print 'class survival rate', df[df.Pclass==2].Survived.mean()
	#print 'class survival rate', df[df.Pclass==3].Survived.mean()
	#Pcalss info process

	#embark
	#print 'noEmbarked survival rate', df[df.Embarked=='Missing'].Survived.mean()
	#print 'Embarked survival rate', df[df.Embarked!='Missing'].Survived.mean()
	#for e in 'SCQ':
	#	print 'Embarked %s survival rate'%e, df[df.Embarked==e].Survived.mean()
		#print '# of embareked %s'%e,df[df.Embarked==e].count()


	#plt.show()
def report(grid_scores, n_top=3):
	top_scores = sorted(grid_scores,key=itemgetter(1),reverse=True)[:n_top]
	for i,score in enumerate(top_scores):
		print("Model with rank: {0}".format(i + 1))
		print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
			score.mean_validation_score,
			np.std(score.cv_validation_scores)))
		print("Parameters: {0}".format(score.parameters))
		print("")




df = feature_engineering(train)
'''
formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_per_person+Fare+Family_size' 
y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
y_train = np.asarray(y_trrain).ravel()
'''
y_train = df.Survived
x_train = df.drop(['Survived','SibSp','Parch','Ticket','Embarked','Family','AgeFill','Cabin'],axis=1)

#print y_train.shape,x_train.shape

#split test/train set

'''
X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train,test_size=0.3, random_state = seed)

rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5)

param_grid = dict()

pipeline = Pipeline([('rfc',rfc)])
grid_search = GridSearchCV(pipeline,param_grid=param_grid,verbose=3,scoring='accuracy',\
	cv=StratifiedShuffleSplit(Y_train,n_iter=10,test_size=0.3,random_state=seed)).fit(X_train,Y_train)

print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
report(grid_search.grid_scores_)

print('-----grid search end------------')
print ('on all train set')
scores = cross_val_score(grid_search.best_estimator_, x_train, y_train,cv=3,scoring='accuracy')
print scores.mean(),scores
print ('on test set')
scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test,cv=3,scoring='accuracy')
print scores.mean(),scores

print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train) ))
print('test data')
print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test) ))

#serialize training
model_file='c:/users/jinji/desktop/titanic/model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)
clf = joblib.load(model_file)

print type(clf)
print clf
'''
'''
y_range=[]
x_range=[]
for i in range(1,10):
	print i
	rfc = RandomForestClassifier(n_estimators=i*100, criterion='entropy', max_depth=5,random_state=seed)
	scores = cross_val_score(rfc,x_train,y_train,cv=10,scoring='accuracy')
	x_range.append(i*100)
	y_range.append(scores.mean())
plt.plot(x_range,y_range)
plt.show()
'''
rfc = RandomForestClassifier(n_estimators=500, max_depth=5, bootstrap=False, criterion='entropy',random_state=seed, min_samples_split=1)
'''
param_dict = dict(n_estimators = [500], max_depth = [5])
rand = RandomizedSearchCV(rfc,param_dict,cv=10,verbose=2,scoring='accuracy',n_iter=1,random_state=seed)
rand.fit(x_train,y_train)
print rand.best_score_
print rand.best_params_
print rand.best_estimator_
'''
rfc.fit(x_train,y_train)
joblib.dump(rfc,'c:/users/jinji/desktop/titanic/model-rf.pkl')

