#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectPercentile,f_classif,chi2 ,RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from tester import test_classifier

#function to divide when there is a posibility of a string
def div(num,den):
	if den=='NaN':
		return 0
	elif num=='NaN':
		num=0

	return float(num)/den

#Check how many missing values there for each feature
def check_missing_values(data_dict):
	import pandas
	pd_data= pandas.DataFrame.from_dict(data_dict,orient='index')
	print 'Missing values for each feature'
	for col in pd_data:
		if col!='poi':
			print col,sum(pd_data[col]=='NaN')

def main():
	print 'Starting script...'
	### Task 1: Select what features you'll use.
	### features_list is a list of strings, each of which is a feature name.
	### The first feature must be "poi".
	features_all = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
					'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
					'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','n_fraction_to_POI','n_fraction_from_POI']

	features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
					'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','shared_receipt_with_poi','n_fraction_to_POI','n_fraction_from_POI']

	### Load the dictionary containing the dataset
	with open("final_project_dataset.pkl", "r") as data_file:
	    data_dict = pickle.load(data_file)

	print 'Number of data points:',len(data_dict)
	print 'Number of features:',len(data_dict['TOTAL'])
	print data_dict['TOTAL']

	### Task 2: Remove outliers
	data_dict.pop('TOTAL',0)

	#check how many features have missing values
	# check_missing_values(data_dict)

	### Task 3: Create new feature(s)
	### Store to my_dataset for easy export below.
	for person in data_dict:
		data_dict[person]['n_fraction_to_POI']=div(data_dict[person]['from_this_person_to_poi'],data_dict[person]['to_messages'])
		data_dict[person]['n_fraction_from_POI']=div(data_dict[person]['from_poi_to_this_person'],data_dict[person]['from_messages'])
	my_dataset = data_dict

	#print data_dict['SKILLING JEFFREY K']

	### Extract features and labels from dataset for local testing
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	### Task 4: Try a varity of classifiers
	### Please name your classifier clf for easy export below.
	### Note that if you want to do PCA or other multi-stage operations,
	### you'll need to use Pipelines. For more info:
	### http://scikit-learn.org/stable/modules/pipeline.html

	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

	# Provided to give you a starting point. Try a variety of classifiers.
	scaler= MinMaxScaler()
	clf = RandomForestClassifier(n_estimators = 10, random_state=42,n_jobs=-1)
	# clf = DecisionTreeClassifier(random_state=42)
	# clf = SVC(C=1,kernel='rbf')
	# clf= AdaBoostClassifier(n_estimators=100,random_state=42)
	selector = SelectPercentile(f_classif,percentile=80)
	pipe = Pipeline([('scaler',scaler),
					 ('selector',selector),
					 ('classifier',clf)
					])
	pipe.fit(features_train,labels_train)
	labels_pred = pipe.predict(features_test)

	print 'Precison:',metrics.precision_score(labels_test,labels_pred)
	print 'Recall:',metrics.recall_score(labels_test,labels_pred)
	print 'F1 Score:',metrics.f1_score(labels_test,labels_pred)

	### Task 5: Tune your classifier to achieve better than .3 precision and recall 
	### using our testing script. Check the tester.py script in the final project
	### folder for details on the evaluation method, especially the test_classifier
	### function. Because of the small size of the dataset, the script uses
	### stratified shuffle split cross validation. For more info: 
	### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

	print '\nTunning parameters...'
	
	#For RandomForest
	params=dict(selector__percentile=[25,50,75,100],
				classifier__n_estimators=[10,50,100],
	 			classifier__criterion=['gini','entropy'])

	#For AdaBoost:
	# params=dict(selector__percentile=[25,50,75,100],
	# 			classifier__n_estimators=[10,50,100],
	# 			classifier__algorithm =['SAMME.R','SAMME'])

	cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)

	param_tuner = GridSearchCV(pipe,params,scoring='f1',cv=cv)
	param_tuner.fit(features,labels)

	print 'Best params:',param_tuner.best_params_
	pipe = param_tuner.best_estimator_
	test_classifier(pipe, data_dict, features_list)

	### Task 6: Dump your classifier, dataset, and features_list so anyone can
	### check your results. You do not need to change anything below, but make sure
	### that the version of poi_id.py that you submit can be run on its own and
	### generates the necessary .pkl files for validating your results.

	dump_classifier_and_data(pipe, my_dataset, features_list)
	print 'End of script'

if __name__ == '__main__':
	main()