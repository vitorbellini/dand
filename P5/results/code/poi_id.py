#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
import numpy as np
import pprint
import matplotlib
from time import time
from collections import defaultdict
from sklearn.metrics import recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 
 'total_payments', 'exercised_stock_options', 
 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 
 'restricted_stock_deferred', 'total_stock_value', 'expenses', 
 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 
 'director_fees', 'deferred_income', 'long_term_incentive', 
 'email_address', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# find out features in the dataset
print "Features available in the dataset: "
for person in data_dict:
    print data_dict[person].keys()
    print len(data_dict[person].keys())
    total_features = len(data_dict[person].keys())
    break
print ""

# find out features with lots of NaN values
NaN_dict = defaultdict(int)
count_poi = 0
count_non_poi = 0

for person in data_dict:
    for key, value in data_dict[person].iteritems():
        if value == 'NaN':
            NaN_dict[key] += 1
        if key == 'poi':
            if value == True:
                count_poi += 1
            elif value == False:
                count_non_poi += 1

print "count keys with NaN values:"
pprint.pprint(sorted( ((v,k) for k,v in NaN_dict.iteritems()), reverse=True))
print ""

# dataset shape
print "total data points: ", len(data_dict)
print "number of features: ", total_features
print "number of POIs / non-POIs: {0} / {1}".format(count_poi, count_non_poi)
print ""

# investigate high NaN occurrences
high_NaN = []
for key in NaN_dict:
    if NaN_dict[key] / float(len(data_dict)) > 0.6: # more than 60% with NaN
        high_NaN.append(key)

for key in high_NaN:
    print "investigate key {0} with high NaN count --------".format(key)
    print ""
    count_key_pois = 0
    for person in data_dict:
        if data_dict[person][key] != 'NaN':
            print person, data_dict[person][key], "poi > ", \
            data_dict[person]['poi']
            if data_dict[person]['poi']:
                count_key_pois += 1
    print "percentage POIs in this key: ", \
    round(count_key_pois / float(count_poi) * 100, 2), "%"
    print ""

# update features list
# restricted_stock_deferred, director_fees, loan_advances seems to contribute
# low. Has a lot of NaN and the ones with value are from non-POIs
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 
 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
 'shared_receipt_with_poi', 'total_stock_value', 'expenses', 
 'from_messages', 'other', 'from_this_person_to_poi', 'deferred_income', 
 'long_term_incentive', 'from_poi_to_this_person']

### Task 2: Remove outliers

# plot salary bonus
for person in data_dict:
    salary = data_dict[person]['salary']
    bonus = data_dict[person]['bonus']
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.savefig('outlier.png')
matplotlib.pyplot.show()


# find the outlier
for person in data_dict:
    try:
        if int(data_dict[person]['salary']) > 25000000:
            print "outlier: ", person
            outlier = person
    except:
        pass
    
# drop the outlier
data_dict.pop(outlier, 0)

# check if there is person with only one name or more than 4
for person in data_dict:
    name = person.split(" ")
    if len(name) < 2 or len(name) > 4:
        print "outlier: ", person
        outlier = person

# drop the outlier
data_dict.pop(outlier, 0)

# new plot without the outlier
for person in data_dict:
    salary = data_dict[person]['salary']
    bonus = data_dict[person]['bonus']
 #   color = "r" if data_dict[person]['poi'] else "b"
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.savefig('outlier_fix.png')
matplotlib.pyplot.show()

### Task 3: Create new feature(s)

# plot to and from messages with poi
for person in data_dict:
    to_poi = data_dict[person]['from_this_person_to_poi']
    from_poi = data_dict[person]['from_poi_to_this_person']
    color = "r" if data_dict[person]['poi'] else "b"
    matplotlib.pyplot.scatter( to_poi, from_poi, c=color,  )

matplotlib.pyplot.xlabel("to_poi")
matplotlib.pyplot.ylabel("from_poi")
matplotlib.pyplot.show()

# create new features
for person in data_dict:
    try:
        data_dict[person]['fraction_to_poi'] = \
        int(data_dict[person]['from_this_person_to_poi']) \
        / float(data_dict[person]['from_messages'])
    except:
        data_dict[person]['fraction_to_poi'] = "NaN"
    try:
        data_dict[person]['fraction_from_poi'] = \
        int(data_dict[person]['from_poi_to_this_person']) \
        / float(data_dict[person]['to_messages'])
    except:
        data_dict[person]['fraction_from_poi'] = "NaN"

# check result. if fractions are below 1
fraction_keys = ['fraction_to_poi', 'fraction_from_poi']

for fraction_key in fraction_keys:
    print "assess {0} result ----------------".format(fraction_key)
    for person in data_dict:
        if data_dict[person][fraction_key] > 1 \
        and data_dict[person][fraction_key] != "NaN":
            print person
            print data_dict[person]
    print ""

# plot the result
for person in data_dict:
    fraction_to_poi = data_dict[person]['fraction_to_poi']
    fraction_from_poi = data_dict[person]['fraction_from_poi']
    color = "r" if data_dict[person]['poi'] else "b"
    matplotlib.pyplot.scatter( fraction_to_poi, fraction_from_poi, c=color,  )

matplotlib.pyplot.xlabel("fraction_to_poi")
matplotlib.pyplot.ylabel("fraction_from_poi")
matplotlib.pyplot.show()

# update features_list with new features fraction_to_poi, fraction_from_poi

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 
 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
 'shared_receipt_with_poi', 'total_stock_value', 'expenses', 
 'from_messages', 'other', 'from_this_person_to_poi', 'deferred_income', 
 'long_term_incentive', 'from_poi_to_this_person', 'fraction_to_poi', 
 'fraction_from_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Use decision tree and selectKBest to identify feature importance
# Decision tree feature importances
#clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(criterion='gini', max_features=0.75, \
                             min_samples_leaf=3, min_samples_split=2)
clf.fit(features, labels)

print "Decision Tree feature importances:"
importances = clf.feature_importances_
for i in range(len(importances)):
    print importances[i], features_list[i + 1]

print ""

# SelectKBest feature importances
selector = SelectKBest(k='all')
selector.fit(features, labels)

print "SelectKBest feature scores:"
scores = selector.scores_
for i in range(len(scores)):
    print scores[i], features_list[i + 1]

print ""

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# split the data to test and train
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Test several feature preparations and classifiers
def try_classifier(estimators, param_grid, classifier_results, classifier):
    """get pipeline estimators and parameters and apply
    GridSearchCV to assess the algorithm prediction results
    
    Args:
        - estimators: pipeline steps
        - param_grid: dictionary with algorithms parameters to test
        - classifier_results: dictionary to append the classifier results
        - classifier: string with the classifier specs.
    Return:
        - classifier_results: dictionary with assessed results of the classifier
    """
    
    classifier_results[classifier] = {}
    pipe = Pipeline(estimators)
    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring='recall')
    t0 = time()
    grid_search.fit(features_train, labels_train)
    classifier_results[classifier]['tempo de treinamento'] = round(time()-t0, 3)
    classifier_results[classifier]['best score'] = \
    round(grid_search.best_score_, 3)
    best_parameters = grid_search.best_estimator_.get_params()
    best_param_list = {}
    for param_name in sorted(param_grid.keys()):
        best_param_list[param_name] = best_parameters[param_name]
        
    classifier_results[classifier]['best parameters'] = best_param_list
    t0 = time()
    predictions = grid_search.predict(features_test)
    classifier_results[classifier]['tempo de previsao'] = round(time()-t0, 3)
    classifier_results[classifier]['precision_score'] = \
    round(precision_score(labels_test, predictions), 3)
    classifier_results[classifier]['recall_score'] = \
    round(recall_score(labels_test, predictions), 3)
    return classifier_results
   
classifier_results = {}

# feature selection: None, classifier: GaussianNB
classifier = 'fs: None, c: GaussianNB'
estimators = [('clf', GaussianNB())]
param_grid = dict()
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature selection: Select Percentile, classifier: GaussianNB
classifier = 'fs: SelectPercentile, c: GaussianNB'
estimators = [('reduce_dim', SelectPercentile()), ('clf', GaussianNB())]
param_grid = dict(reduce_dim__percentile=range(10, 50, 5))
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature selection: PCA, classifier: GaussianNB
classifier = 'fs: PCA, c: GaussianNB'
estimators = [('reduce_dim', PCA()), ('clf', GaussianNB())]
param_grid = dict(reduce_dim__n_components=range(2, len(features[0]), 1))
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature prep: MinMaxScaler, feature selection: None, classifier: GaussianNB
classifier = 'fp: MinMax, fs: None, c: GaussianNB'
estimators = [('prepare_features', MinMaxScaler()), ('clf', GaussianNB())]
param_grid = dict()
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature prep: MinMaxScaler, feature selection: Select Percentile, classifier: GaussianNB
classifier = 'fp: MinMax, fs: SelectPercentile, c: GaussianNB'
estimators = [('prepare_features', MinMaxScaler()), ('reduce_dim', SelectPercentile()), ('clf', GaussianNB())]
param_grid = dict(reduce_dim__percentile=range(10, 50, 5))
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature prepfeature selection: PCA, classifier: GaussianNB
classifier = 'fp: MinMax, fs: PCA, c: GaussianNB'
estimators = [('prepare_features', MinMaxScaler()), ('reduce_dim', PCA()), ('clf', GaussianNB())]
param_grid = dict(reduce_dim__n_components=range(2, len(features[0]), 1))
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature prep: MinMax, feature selection: None, classifier: SVC
classifier = 'fp: MinMax, fs: None, c: SVC'
estimators = [('prepare_features', MinMaxScaler()), ('clf', SVC())]
param_grid = dict(clf__C=[0.1, 1, 10, 100, 1000, 10000],
                  clf__verbose=[True, False])
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature prep: MinMax, feature selection: Select Percentile, classifier: SVC
classifier = 'fp: MinMax, fs: Select Percentile, c: SVC'
estimators = [('prepare_features', MinMaxScaler()),('reduce_dim', \
              SelectPercentile()), ('clf', SVC())]
param_grid = dict(reduce_dim__percentile=range(10, 50, 5),
                  clf__C=[0.1, 1, 10, 100, 1000, 10000],
                  clf__verbose=[True, False])
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature prep: MinMax, feature selection: PCA, classifier: SVC
classifier = 'fp: MinMax, fs: PCA, c: SVC'
estimators = [('prepare_features', MinMaxScaler()),('reduce_dim', \
              PCA()), ('clf', SVC())]
param_grid = dict(reduce_dim__n_components=range(2, len(features[0]), 1),
                  clf__C=[0.1, 1, 10, 100, 1000, 10000],
                  clf__verbose=[True, False])
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# using for Decision Tree the highest feature importances as input
features_list = ['poi', 'other', 'exercised_stock_options', 
                 'expenses', 'fraction_to_poi',
                 'shared_receipt_with_poi']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Train and test with split train and test data
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


# feature selection: None, classifier: Decision Tree
classifier = 'fs: None, c: Decision Tree'
estimators = [('clf', DecisionTreeClassifier(random_state=42))]
param_grid = dict(clf__min_samples_split=range(2, 6, 1),
                  clf__max_depth=range(2,6,1),
                  clf__min_samples_leaf=[1, 2, 3],
                  clf__criterion=['gini', 'entropy'],
                  clf__class_weight=['balanced'])
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature selection: Select Percentile, classifier: Decision Tree
classifier = 'fs: Select Percentile, c: Decision Tree'
estimators = [('reduce_dim', SelectPercentile()), ('clf', \
              DecisionTreeClassifier(random_state=42))]
param_grid = dict(reduce_dim__percentile=range(10, 50, 5),
                  clf__min_samples_split=range(2, 6, 1),
                  clf__max_depth=range(2,6,1),
                  clf__min_samples_leaf=[1, 2, 3],
                  clf__criterion=['gini', 'entropy'],
                  clf__class_weight=['balanced'])
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)


# feature prep: MinMax, feature selection: PCA, classifier: GaussianNB
classifier = 'fs: PCA, c: Decision Tree'
estimators = [('reduce_dim', PCA()), ('clf', \
              DecisionTreeClassifier(random_state=42))]
param_grid = dict(reduce_dim__n_components=range(2, len(features[0]), 1),
                  clf__min_samples_split=range(2, 6, 1),
                  clf__max_depth=range(2,6,1),
                  clf__min_samples_leaf=[1, 2, 3],
                  clf__criterion=['gini', 'entropy'],
                  clf__class_weight=['balanced'])
classifier_results = \
try_classifier(estimators, param_grid, classifier_results, classifier)

# Check the results
pprint.pprint(classifier_results)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# features list. decision tree features importance higher than 0.1
features_list = ['poi', 'other', 'exercised_stock_options', 
                 'expenses', 'fraction_to_poi',
                 'shared_receipt_with_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Train and test with split train and test data
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# classify
clf = DecisionTreeClassifier(criterion='gini', class_weight='balanced', \
                             max_depth=2, min_samples_leaf=1, \
                             min_samples_split=2)
clf.fit(features_train, labels_train)

# test
pred = clf.predict(features_test)
print ""
print "Classification report for Decision Tree with train/test split 0.3:"
print classification_report(labels_test, pred)


### Train and Test with cross validation kfold
cross_validation_recall_results = []
cross_validation_precision_results = []
kf = KFold(len(features), 4, shuffle=True)
for train_indices, test_indices in kf:
    # make training and testing datasets
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]
    
    # classify
    clf = DecisionTreeClassifier(criterion='gini', class_weight='balanced', \
                             max_depth=2, min_samples_leaf=1, \
                             min_samples_split=2)
    clf.fit(features_train, labels_train)

    # test
    pred = clf.predict(features_test)
    cross_validation_recall_results.append(recall_score(labels_test, pred))
    cross_validation_precision_results.append(precision_score(labels_test, pred))

print "Classification report for Decision Tree with KFold k=4"
print "recall mean: ", np.mean(cross_validation_recall_results)
print "precision mean: ", np.mean(cross_validation_precision_results)

# test with test_classifier
clf = DecisionTreeClassifier(criterion='gini', class_weight='balanced', \
                             max_depth=2, min_samples_leaf=1, \
                             min_samples_split=2)
print ""
print "test_classifier for test.py"
test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)