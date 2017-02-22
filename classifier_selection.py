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

def div(num,den):
	'''function to divide when there is a posibility of a string'''
	if den=='NaN':
		return 0
	elif num=='NaN':
		num=0

	return float(num)/den

def check_missing_values(data_dict):
	'''Check how many missing values there for each feature'''
	import pandas
	pd_data= pandas.DataFrame.from_dict(data_dict,orient='index')
	print 'Missing values for each feature'
	for col in pd_data:
		if col!='poi':
			print col,sum(pd_data[col]=='NaN')

def main():
	print 'Starting script...'
	
	# List of features that will be used
	features_all = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
					'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
					'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','n_fraction_to_POI','n_fraction_from_POI']

	features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
					'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','shared_receipt_with_poi','n_fraction_to_POI','n_fraction_from_POI']

	# Load the dictionary containing the dataset. Dataset created using a different Python script (\tools)
	with open("final_project_dataset.pkl", "r") as data_file:
	    data_dict = pickle.load(data_file)

	# Dimentions of the data
	print 'Number of data points:',len(data_dict)
	print 'Number of features:',len(data_dict['TOTAL'])
	print data_dict['TOTAL']

	# Remove outlier
	data_dict.pop('TOTAL',0)

	#check how many features have missing values
	# check_missing_values(data_dict)

	# Add two new features
	for person in data_dict:
		data_dict[person]['n_fraction_to_POI']=div(data_dict[person]['from_this_person_to_poi'],data_dict[person]['to_messages'])
		data_dict[person]['n_fraction_from_POI']=div(data_dict[person]['from_poi_to_this_person'],data_dict[person]['from_messages'])
	my_dataset = data_dict

	# Extract features and labels from dataset for local testing
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	# Split data into train and test sets
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

	# Min Max scaler for features
	scaler= MinMaxScaler()

	# Try differnt classifiers. RandomForest had the highest F1 score.
	clf = RandomForestClassifier(n_estimators = 10, random_state=42,n_jobs=-1)
	# clf = DecisionTreeClassifier(random_state=42)
	# clf = SVC(C=1,kernel='rbf')
	# clf= AdaBoostClassifier(n_estimators=100,random_state=42)

	# Feature selector
	selector = SelectPercentile(f_classif,percentile=80)

	# Pipeline to combine scaler, selector and classifier together
	pipe = Pipeline([('scaler',scaler),
					 ('selector',selector),
					 ('classifier',clf)
					])

	# Fit and make classifier prediction
	pipe.fit(features_train,labels_train)
	labels_pred = pipe.predict(features_test)

	print 'Precison:',metrics.precision_score(labels_test,labels_pred)
	print 'Recall:',metrics.recall_score(labels_test,labels_pred)
	print 'F1 Score:',metrics.f1_score(labels_test,labels_pred)

	# Parameter tuning
	print '\nTunning parameters...'
	
	# Parameter dictionary for RandomForest
	params=dict(selector__percentile=[25,50,75,100],
				classifier__n_estimators=[10,50,100],
	 			classifier__criterion=['gini','entropy'])

	# Parameter dictionary for AdaBoost:
	# params=dict(selector__percentile=[25,50,75,100],
	# 			classifier__n_estimators=[10,50,100],
	# 			classifier__algorithm =['SAMME.R','SAMME'])

	# Cross validation function
	cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)

	# Execute paramter turning and fit data to the best model
	param_tuner = GridSearchCV(pipe,params,scoring='f1',cv=cv)
	param_tuner.fit(features,labels)

	print 'Best params:',param_tuner.best_params_
	pipe = param_tuner.best_estimator_

	# Test classifier
	test_classifier(pipe, data_dict, features_list)

	# Dump classifier and data for evaluation
	dump_classifier_and_data(pipe, my_dataset, features_list)
	print 'End of script'

if __name__ == '__main__':
	main()