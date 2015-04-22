#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    if poi_messages == "NaN" or poi_messages == "nan":
        fraction = 0.
    elif all_messages != 0:
        fraction = float(poi_messages) / float(all_messages)
    else:
        fraction = 0.
    return fraction

def scaleFinancialFeatures(dataset, financial_feature_list):
    """ given a dataset, the list of all features, and the list of financial
        features, scale the financial features in such a way that 0.0 values
        are ignored.
    """ 
    for f in financial_feature_list:
        vmin = sys.float_info.max
        vmax = -sys.float_info.max
        for name in data_dict:
            vals = data_dict[name]
            val = float(vals[f])
            if not np.isnan(val):
                if vmin > val:
                    vmin = val
                if vmax < val:
                    vmax = val
        # print f," ",vmin," ",vmax
        if vmin < vmax:
            for name in data_dict:
                if not np.isnan(val):
                    val = float(data_dict[name][f])
                    # Why is this necessary?  Python bug?
                    if not np.isnan(val):
                        data_dict[name][f] = (val - vmin ) / ( vmax - vmin )

def scaleEmailFeatures(dataset, email_feature_list):
    """ given a dataset and the list of email features, scale the features
    """ 
    for f in email_feature_list:
        vmin = sys.float_info.max
        vmax = -sys.float_info.max
        for name in data_dict:
            vals = data_dict[name]
            val = vals[f]
            if val != "NaN":
                if vmin > val:
                    vmin = val
                if vmax < val:
                    vmax = val
        # print f," ",vmin," ",vmax
        if vmin < vmax:
            for name in data_dict:
                val = data_dict[name][f]
                if val != "NaN" and val != "nan":
                    data_dict[name][f] = ( val - vmin ) / ( vmax - vmin )

def checkScaling(dataset, feature_list):
    """ given a dataset check scaling
    """ 
    for f in feature_list:
        vmin = sys.float_info.max
        vmax = -sys.float_info.max
        if f != "poi":
            for name in data_dict:
                vals = data_dict[name]
                val = vals[f]
                if val != "NaN":
                    if vmin > val:
                        vmin = val
                    if vmax < val:
                        vmax = val
            print f," ",vmin," ",vmax

# Automatic selection of good features for classifiers using SelectPercentile.
def selectFeatures(dataset, feature_list):
    """ Given a data set and a list of features, use SelectPercentile to find 
        the most effective features for solving the classification problem.
    """
    data = featureFormat(dataset, feature_list, sort_keys = True) 
    labels, features = targetFeatureSplit(data)
    from sklearn.feature_selection import f_classif,SelectPercentile
    # answers were insensitive to percentile
    selector = SelectPercentile(f_classif, percentile=25)
    selector.fit(features, labels)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    # Leave out poi from the list of features
    for fl,s in zip(feature_list[1:],scores):
        print fl,':\t',s

def addStockFractions(data_dict):
    """ Create and populate the new columns 'frac_exercised_stock_options',
        'frac_restricted_stock' and 'frac_restricted_stock_deferred'.
    """
    unnormalizeds = ['exercised_stock_options', 'restricted_stock_deferred', 'restricted_stock']
    for name in data_dict:
        vals = data_dict[name]
        for vi in unnormalizeds:
            vo = 'frac_' + vi
            if vals[vi] != "NaN" and vals["total_stock_value"] != "NaN":
                vals[vo] = vals[vi] / vals["total_stock_value"]
            else:
                vals[vo] = "NaN"
 
def addPaymentFractions(data_dict):
    """ Create and populate the new columns 'frac_salary', 'frac_bonus', 'frac_deferred_income',
        'frac_expenses' and 'frac_other', and 'frac_long_term_incentive'.
    """
    unnormalizeds = ['salary', 'bonus', 'deferral_payments', 'loan_advances', 'deferred_income',
        'expenses', 'other', 'long_term_incentive']
    for name in data_dict:
        vals = data_dict[name]
        for vi in unnormalizeds:
            vo = 'frac_' + vi
            if vals[vi] != "NaN" and vals["total_payments"] != "NaN":
                vals[vo] = float(vals[vi]) / float(vals["total_payments"])
            else:
                vals[vo] = "NaN"

def transformNanTotalPayments(data_dict):
    unnormalizeds = ['salary', 'bonus', 'deferral_payments', 'loan_advances', 'deferred_income',
        'expenses', 'other', 'long_term_incentive']
    for name in data_dict:
        vals = data_dict[name]
        pt = 0.0
        if vals["total_payments"] == "NaN":
            for k in  unnormalizeds:
                if vals[k] != "NaN":
                    pt = pt + vals[k]
            vals["total_payments"] = pt

def transformNanTotalStockValue(data_dict):
    unnormalizeds = ['exercised_stock_options', 'restricted_stock_deferred', 'restricted_stock']
    for name in data_dict:
        vals = data_dict[name]
        pt = 0.0
        if vals["total_stock_value"] == "NaN":
            for k in  unnormalizeds:
                if vals[k] != "NaN":
                    pt = pt + vals[k]
            vals["total_stock_value"] = pt

def addEmailFractions(data_dict):
    """ Create and populate the new columns 'fraction_to_poi', 'fraction_from_poi', 'frac_shared_receipt_with_poi'
    """
    for name in data_dict:
        vals = data_dict[name]

        from_this_person_to_poi = vals["from_this_person_to_poi"]
        from_messages = vals["from_messages"]
        fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
        vals["fraction_to_poi"] = fraction_to_poi

        from_poi_to_this_person = vals["from_poi_to_this_person"]
        to_messages = vals["to_messages"]
        fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
        vals["fraction_from_poi"] = fraction_from_poi

        shared_receipts = vals["shared_receipt_with_poi"]
        fraction_shared_receipts = computeFraction(shared_receipts, to_messages)
        vals["frac_shared_receipt_with_poi"] = fraction_shared_receipts

def transformNansToMedian(data_dict, column_list):
    """ Replace NaN and nan entries with column medians
    """
    for k in column_list:
        vs = []
        for name in data_dict:
            v = data_dict[name][k]
            if v != "NaN" and v != "nan":
                vs.append(v)
        vm = np.median(vs)
        for name in data_dict:
            v = data_dict[name][k]
            if v == "NaN" or v == "nan":
                data_dict[name][k] = vm

def transformNansToZero(data_dict, column_list):
    """ Replace NaN and nan entries with zeros
    """
    for k in column_list:
        for name in data_dict:
            v = data_dict[name][k]
            if v == "NaN" or v == "nan":
                data_dict[name][k] = 0.0

def findSVCParams(features, labels):
    """ Find a locally best choice for gamma and C, parameters for SVM.
    """
    from sklearn.cross_validation import StratifiedShuffleSplit
    from sklearn.grid_search import GridSearchCV
    C = 129154.96650148826
    gamma = 1.9306977288832455e-18
    bestCGamma = dict(gamma=gamma, C=C)
    C_range = np.logspace(0, 3, 32)
    gamma_range = np.logspace(0, 3, 32)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(labels, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    grid.fit(features, labels)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    return grid.best_params_

def dumpDeferralPayments(data_dict):
    for name in data_dict:
        if data_dict[name]['deferral_payments'] != "NaN":
            print name," ",data_dict[name]

def dumpDeferredIncome(data_dict):
    for name in data_dict:
        if data_dict[name]['deferred_income'] != "NaN":
            print name," ",data_dict[name]

def dumpLoanAdvances(data_dict):
    for name in data_dict:
        if data_dict[name]['loan_advances'] != "NaN":
            print name," ",data_dict[name]

def dumpVals(data_dict, k):
    for name in data_dict:
        if data_dict[name][k] != "NaN":
            print name," ",data_dict[name][k]

def countPoi(data_dict):
    nameCount = 0
    poiCount = 0
    for name in data_dict:
        nameCount = nameCount + 1
        if data_dict[name]['poi'] == 1:
            poiCount = poiCount + 1
    print "Number of rows: ",nameCount," Number of pois: ",poiCount

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Accuracy: 0.86667     Precision: 0.50000      Recall: 0.396000
features_list = ['poi', 'exercised_stock_options', 'salary', 'bonus', 'deferred_income', 'long_term_incentive' ]
# Accuracy: 0.81117	Precision: 0.30030	Recall: 0.10000
# features_list = ['poi', 'frac_bonus', 'fraction_to_poi', 'frac_long_term_incentive', 'frac_shared_receipt_with_poi' ]
# Accuracy: 0.85613	Precision: 0.45117	Recall: 0.36500
# features_list = ['poi', 'exercised_stock_options', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income', 'long_term_incentive' ]
# Accuracy: 0.86220	Precision: 0.47713	Recall: 0.34950
# features_list = ['poi', 'exercised_stock_options', 'bonus', 'salary', 'fraction_to_poi', 'deferred_income']

# SVC used scaling.  It never found a full set of
# features_list = ['poi', 'exercised_stock_options']
# features_list = ['poi', 'exercised_stock_options', 'salary' ]
# features_list = ['poi', 'exercised_stock_options', 'salary', 'bonus']
# features_list = ['poi', 'exercised_stock_options', 'salary', 'bonus', 'deferred_income', 'long_term_incentive' ]

# raw_medianbase_financial_features_list = ['salary', 'bonus', 'expenses']
raw_medianbase_financial_features_list = []

raw_zerobase_financial_features_list = ['salary', 'bonus', 'expenses', 'deferral_payments', 'loan_advances',
'restricted_stock_deferred', 'deferred_income', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock']

# new_medianbase_financial_features_list = ['frac_salary', 'frac_bonus', 'frac_expenses']
new_medianbase_financial_features_list = []

new_zerobase_financial_features_list = ['frac_salary', 'frac_bonus', 'frac_expenses',
'frac_deferral_payments', 'frac_loan_advances',
'frac_restricted_stock_deferred', 'frac_deferred_income', 'frac_exercised_stock_options',
'frac_other', 'frac_long_term_incentive', 'frac_restricted_stock']

raw_financial_features_list = raw_medianbase_financial_features_list + raw_zerobase_financial_features_list
new_financial_features_list = new_medianbase_financial_features_list + new_zerobase_financial_features_list

raw_email_features_list = ['from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']

new_email_features_list = ['fraction_from_poi', 'fraction_to_poi', 'frac_shared_receipt_with_poi']

raw_features_list = raw_financial_features_list + raw_email_features_list
new_features_list = new_financial_features_list + new_email_features_list
mixed_features_list = raw_financial_features_list + new_email_features_list

rawnew_features_list = raw_features_list + new_features_list

#features_list = ['poi', 'salary']
#features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus',
#       'salary','deferred_income','long_term_incentive', 'total_payments']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# countPoi(data_dict)
# dumpLoanAdvances(data_dict)
# dumpDeferralPayments(data_dict)
# dumpVals(data_dict,'restricted_stock_deferred')
# dumpVals(data_dict,'restricted_stock')
# print 'BHATNAGAR SANJAY'
# print data_dict['BHATNAGAR SANJAY']
# print 'BELFER ROBERT'
# print data_dict['BELFER ROBERT']

### Task 2: Remove outliers

data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)

addStockFractions(data_dict)
addPaymentFractions(data_dict)
addEmailFractions(data_dict)

### Fill in NaNs with zero or median values

transformNanTotalPayments(data_dict)
transformNanTotalStockValue(data_dict)

#dumpVals(data_dict,'frac_salary')

transformNansToZero(data_dict, raw_zerobase_financial_features_list + ['director_fees'])
transformNansToMedian(data_dict, raw_email_features_list)

transformNansToZero(data_dict, new_zerobase_financial_features_list)
transformNansToMedian(data_dict, new_email_features_list)

### Scale the data or not

scaleFinancialFeatures(data_dict, raw_financial_features_list + ['director_fees'])
scaleEmailFeatures(data_dict, raw_email_features_list)
scaleFinancialFeatures(data_dict, new_financial_features_list)
scaleEmailFeatures(data_dict, new_email_features_list)
scaleFinancialFeatures(data_dict, ['director_fees'])

### Find the best features to include for classification
# print "RAW"
# selectFeatures(data_dict, ['poi'] + raw_features_list + ['director_fees'])
# print "SCALED"
# selectFeatures(data_dict, ['poi'] + new_features_list + ['director_fees'])
# print "MIXED"
# selectFeatures(data_dict, ['poi'] + mixed_features_list + ['director_fees'])
# print "DONE"

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#from sklearn.preprocessing import Imputer
#imp = Imputer(strategy='median',copy=False)
#imp.fit_transform(features)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

# from sklearn.svm import SVC
# checkScaling(data_dict, features_list)
# bestCGamma = findSVCParams(features, labels)
# clf = SVC(kernel='rbf',gamma=bestCGamma['gamma'], C=bestCGamma['C'])

# from sklearn.tree import DecisionTreeClassifier
# print features_list
# clf = DecisionTreeClassifier(criterion='gini')

# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# bestCGamma = findSVCParams(features, labels)
# estimators = [('reduce_dim', PCA()), 
#      ('svm', SVC(kernel='rbf',gamma=bestCGamma['gamma'],C=bestCGamma['C']))]
# clf = Pipeline(estimators)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### For use with DecisionTreeClassifier
# print clf.feature_importances_

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
