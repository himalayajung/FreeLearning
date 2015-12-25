#! /usr/bin/env python

# -*- coding: utf-8 -*-
"""

TO DO:
1) clean 
2) make the code more modular
----

Created on Sat May  9 16:25:50 2015

@author: gajendrakatuwal

================================================
Feature Selecion + Classification Pipeline
================================================

usage: pipeline.py  [-h]
                    [-f feature_selection]
                    [-c classification]
                    [-a attribute]
                    [-s site]
                    [-nf n_folds]
                    [-fn fold_no]
                    [-nt n_tree]
                    data_file (FS.csv)

Examples:
python pipeline.py FSS.csv  -c RF  -a ALL -nt 5000  --scoring roc_auc  -sex 1 --search grid -folder ALL/roc_auc/grid/high_cor_not_removed/nt_5000 -remove_correlated 1  &
"""

import warnings
from sklearn.utils import ConvergenceWarning
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression, RandomizedLasso, LassoCV, LassoLarsCV, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import cross_val_score, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, ExtraTreesClassifier
from scipy.stats import randint as sp_randint
from sklearn.metrics import classification_report, roc_auc_score, make_scorer, accuracy_score
import xgboost as xgb
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
import cPickle as pickle
import os
import sys
import re
from time import time
import logging
# from skll import kappa # not available for scikit-learn 0.16.1
import boruta
from scipy import stats
from unbalanced_dataset import SMOTE, SMOTETomek
from smote import SMOTE
from python_gems import get_stats

# Helper Functions
def remove_highly_correlated(df, method='pearson', threshold=0.95):
    """
    remove highly correlated features before classification
    """
    df_corr = df.corr(method=method)
    # create a mask to ignore self-
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr
    drops = []
    # loop through each variable
    for col in df_corr.columns.values:
        # if we've already determined to drop the current variable, continue
        if np.in1d([col], drops):
           continue
        # find all the variables that are highly correlated with the current variable 
        # and add them to the drop list 
        corr = df_corr[abs(df_corr[col]) > threshold].index
        drops = np.union1d(drops, corr)
    logger.info("Dropping {} highly correlated features...".format(drops.shape[0]))
    return (df.drop(drops, axis=1))

def None_to_str(x, prefix="", suffix=""):
    return '' if x is None else ("_" + prefix + str(x) + suffix)

def None_to_str1(x, prefix="", suffix=""):
    return '' if x is (None or False) else ("_" + prefix + suffix)

def None_to_str2(x, prefix="", suffix=""):
    return '' if x is (None or False) else str(x) + suffix

def one_fold(y):
    """ no splitting of the data; handy to check data leak """
    return zip([np.arange(len(y))], [np.arange(len(y))])

# Custom scoring functions
roc_auc_weighted = make_scorer(roc_auc_score, average='weighted')

# Command line arguments parsing
desc = 'Feature Selecion + Classification Pipeline'
parser = ArgumentParser(description=desc)

parser.add_argument('data_file', help='data_file in .csv or .pkl format')

parser.add_argument('-f', dest='feature_selection', help='feature_selection method')

parser.add_argument('-r', dest='regression', help='regression method')

parser.add_argument('-c', dest='classification', help='classification method')

parser.add_argument('--scoring', '-sc', default="accuracy", help="options are 'accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'roc_auc' ....")

parser.add_argument('--validation', '-vd', default="skf", help="validation type: skf or loo")

parser.add_argument('--attribute', '-a', help='atttribute of data  e.g. _volume, _area or sub_cortical_structure')

parser.add_argument('--site', '-s', help='scanning site to be processed')

parser.add_argument('--n_folds', '-nf', help='No. of folds for cross_validation. Default is 10', type=int, default=10)

parser.add_argument('--fold_no', '-fn', type=int, help='Fold # to use')

parser.add_argument('--n_tree', '-nt', type=int, help='No. of trees for RF, GBM and xgboost. Default is 5000', default=2000)

parser.add_argument('--subsample', '-ss', type=float, default=0.5, help='subsample')

parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate for gradient boosting trees')

parser.add_argument('--use_saved_feature_selection', '-usf', action="store_true")

parser.add_argument('-alpha')

parser.add_argument('--selection_threshold', '-st', type=float, default=0.2, help="selection_threshold for feature_selection in randomized_lasso")

parser.add_argument('--donot_normalize_volume', '-dnv', action='store_true')

parser.add_argument('--save_everything', '-se', action='store_true', help="True if you want to save everything")

parser.add_argument('--seed', '-sd', default=0, type=int)

parser.add_argument('--step_RFE', default=1, type=float, help='step size to reduced no. of features in RFE')

parser.add_argument('--search', default="grid", help="Hyperparameter search type")

parser.add_argument('--n_iter_random_search', '-n_iter', default=20, type=int, help="no of iterations for random hyperparameter search")

parser.add_argument('-ADOS', type=float, nargs='*', help="autism group. 1: ADOS<10, 2: ADOS=10-14, 3: ADOS>14")

parser.add_argument('-AS', type=float, nargs='*', help="ADOS severity")

# parser.add_argument('--age_group', '-age', type=int, help="Age group. 1:age<13, age>=13")

parser.add_argument('-VIQ', type=float, nargs='*', help="VIQ group. 1:VIQ<=85, 2:115>VIQ>85, 3: VIQ>115")

parser.add_argument('--percentile', '-per', type=int, default=90, help="percentile for boruta feature_selection")

parser.add_argument('--save_folder', '-folder', help="folder name to save the results")

parser.add_argument('-age', type=float, nargs='*')

parser.add_argument('-sex', type=int, help="male=1, female=2")

parser.add_argument('-stats', action="store_true", help="if true: returns mean, std, cohhens d, tval, pval")

parser.add_argument('-leak', action="store_true")  # to check data leak due to feature selection

parser.add_argument('-tg', '--tree_grid', nargs='*', type=int, help="grid to search tree, start, end , step")

parser.add_argument('--include_DB', '-db', nargs='*', help="include DB measures age, sex, IQs, site")

parser.add_argument('--balance_classes', '-bc', default="auto", help="create_balanced_classes for ASD and TDC. auto: Weights associated with classes in the form {class_label: weight}")

parser.add_argument('--only_normal_IQ', '-niq', action="store_true", help="only use TDC subjects with normal VIQ while subsetting using ADOS")

parser.add_argument('-mg', '--match_group', default=0, type=int, help="match group. e.g. 1: AS=[0,5]")

parser.add_argument('-mt', '--match_type', default=0, type=int, help="match type. e.g. 1: match by age, 1: match by age+VIQ")

parser.add_argument('-impute', action = "store_true")

parser.add_argument('-remove_correlated', type=float, default=1, help="correlation coeff cuttoff to remove the highley correlated features")

args = parser.parse_args()
# argsd = {k: '' if v is None else v for k, v in vars(args).items()}

# Logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# check the projects and assign appropriate project folder for saving runs
if "zernike" in args.data_file:
    project = 'vertex/'
elif "FS" in args.data_file:
    project = 'freesurfer_classification/'
else:
    pass

folder = project + "saved_runs/" + None_to_str2(args.save_folder, suffix="/")

if not os.path.exists(folder):
    os.makedirs(folder)
    logger.info("{} folder created and everything will be saved under it".format(folder))
    logger.info("SEED: {} initialized".format(args.seed))

# for saving the results
filename = (None_to_str(args.feature_selection) + None_to_str(args.classification) +
            None_to_str(args.attribute) +
            None_to_str(args.sex, 'sex') +
            None_to_str(args.ADOS, 'ADOS') +
            None_to_str(args.AS, 'AS') +
            None_to_str(args.age, 'age') +
            None_to_str(args.VIQ, 'VIQ') +
            None_to_str(args.site) +
            None_to_str(args.fold_no) +
            None_to_str(args.include_DB, 'DB'))
# full filenames
small_filename = folder + "small_" + filename  # To save few important things to make read/write faster
stats_filename = folder + "stats_" + filename
filename = folder + filename

log_file = filename + ".log"
save_file = filename + ".pkl"
small_save_file = small_filename + ".pkl"
file_log_handler = logging.FileHandler(log_file, 'w')
stdout_log_handler = logging.StreamHandler()

# add handlers
logger.addHandler(file_log_handler)
logger.addHandler(stdout_log_handler)

# set levels
file_log_handler.setLevel(logging.DEBUG)
stdout_log_handler.setLevel(logging.DEBUG)

# Formatting output
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_log_handler.setFormatter(formatter)
stdout_log_handler.setFormatter(formatter)

#
logger.info("INPUT:")
logger.info(args)
logger.info("Everything will be saved as: {}".format(filename))


# to print
if args.leak:
    logger.info("Feature selection using all the data")
    saved_feature_selection_file = 'LEAK/small_RFECV_linearSVM___'+args.site+'.pkl'
    logger.info("Selected features from {} will be used".format(saved_feature_selection_file))

if args.feature_selection:
    logger.info("%s used for feature_selection ......." % args.feature_selection)
    if args.use_saved_feature_selection:
        saved_feature_selection_file = os.path.join("feature_selection", args.feature_selection+"__"+None_to_str(args.site)+".p")
        # logger.info("Previously saved feature selection data used: from {} ".format(saved_feature_selection_file))

if args.classification:
    logger.info("%s used for classification ......" % args.classification)

if args.regression:
    logger.info("%s used for regression on ADOS score ......." % args.regression)
           
if (not args.feature_selection) and (not args.classification) and (not args.ttest):
    logger.info("Nothing done. Exiting.")
    sys.exit()

if args.attribute:
    attributes = [args.attribute]
else:
    if "zernike" in args.data_file:
        attributes = ['L_Accu', 'R_Accu', 'L_Amyg', 'R_Amyg', 'L_Caud', 'R_Caud', 'L_Hipp', 'R_Hipp', 'L_Pall', 'R_Pall', 'L_Puta', 'R_Puta', 'L_Thal', 'R_Thal']
        # BrStem has problem
    elif "FS" in args.data_file:
        attributes = ['_volume', '_area', '_thickness$', '_thicknessstd', '_foldind', '_meancurv', '_gauscurv', 'ALL']
    else:
        pass

# Settings
np.random.seed(args.seed)

# Data
data = pd.read_csv(os.path.join('..', 'data', args.data_file), index_col=0)
logger.info(data.columns[:10])  # for sanity check

if args.match_type:
    if args.AS:
        variable_of_interest = 'AS'
    elif args.age:
        variable_of_interest = 'age'
    elif args.VIQ:
        variable_of_interest = 'VIQ'
    elif args.ADOS:
        variable_of_interest = 'ADOS'
    else:
        pass
    logger.info('Matched subjects used from match_subjects.csv. Variable of interst = {}... Match type = {}... Match group = {}.'.format(variable_of_interest, args.match_type, args.match_group))

    mdata = pd.read_csv(os.path.join('..', 'data', 'matched_subjects.csv'), index_col=0)
    mdata = mdata[(mdata['variable_of_interest'] == variable_of_interest) & (mdata['group'] == args.match_group) & (mdata['match_type'] == args.match_type)]

    TDC = data.loc[mdata['controls'], ]
    ASD = data.loc[mdata['cases'], ]
    data = TDC.append(ASD)

    avg = map(lambda x: round(np.mean(x), 2), [data['age'], data['VIQ'], data['ADOS_GOTHAM_SEVERITY']])
    sd = map(lambda x: round(np.std(x), 2), [data['age'], data['VIQ'], data['ADOS_GOTHAM_SEVERITY']])

    logger.info("Mean: age, VIQ, AS = {}".format(avg))
    logger.info("Std.: age, VIQ, AS = {}".format(sd))

    # sys.exit(0)
else:
    if args.sex:
        logger.info('Subjects with sex = {} selected'.format(args.sex))
        data = data[data['sex'] == args.sex]

    if args.age:
        logger.info('Subjects with age >= {} and age <= {} are present in the selected interval'.format(args.age[0], args.age[1]))
        data = data[(data['age'] >= args.age[0]) & (data['age'] <= args.age[1])]
        TDC = data[data['control'] == 1]
        ASD = data[data['control'] == 0]
        logger.info('After subsetting, {} TDCs and {} ASDs  selected'.format(TDC.shape[0], ASD.shape[0]))

    if args.ADOS:
        logger.info('Subjects with ADOS >= {} and ADOS <= {} are present in that interval'.format(args.ADOS[0], args.ADOS[1]))
        TDC = data[data['control'] == 1]
        ASD = data[data['control'] == 0]
        if args.only_normal_IQ:
            TDC = TDC[(TDC['VIQ'] >= 85) & (TDC['VIQ'] <= 115)]# only using normal TDC population
        ASD = ASD[(ASD['ADOS'] >= args.ADOS[0]) & (ASD['ADOS'] <= args.ADOS[1])] # NaN values doesn't affect
        data = TDC.append(ASD)
        logger.info('After subsetting, {} TDCs and {} ASDs  selected'.format(TDC.shape[0], ASD.shape[0]))

    if args.AS:
        logger.info('Subjects with ADOS_SEVERITY >= {} and ADOS_SEVERITY  <= {} are present in that interval'.format(args.AS[0], args.AS[1]))
        TDC = data[data['control'] == 1]
        ASD = data[data['control'] == 0]
        if args.only_normal_IQ:
            TDC = TDC[(TDC['VIQ'] >= 85) & (TDC['VIQ'] <= 115)]# only using normal TDC population
        ASD = ASD[(ASD['ADOS_GOTHAM_SEVERITY'] >= args.AS[0]) & (ASD['ADOS_GOTHAM_SEVERITY'] <= args.AS[1])] # NaN values doesn't affect
        data = TDC.append(ASD)
        logger.info('After subsetting, {} TDCs and {} ASDs  selected'.format(TDC.shape[0], ASD.shape[0]))

    if args.VIQ:
        logger.info('Subjects with VIQ >= {} and VIQ <= {} are present in that interval'.format(args.VIQ[0], args.VIQ[1]))
        data = data[(data['VIQ'] >= args.VIQ[0]) & (data['VIQ'] <= args.VIQ[1])]
        TDC = data[data['control'] == 1]
        ASD = data[data['control'] == 0]
        logger.info('After subsetting, {} TDCs and {} ASDs  selected'.format(TDC.shape[0], ASD.shape[0]))

    if args.site:
        logger.info("{} site will be processed.....".format(args.site))
        data = data[data['site'] == args.site]

    if args.balance_classes:
        TDC = data[data['control'] == 1]
        ASD = data[data['control'] == 0]
        if args.balance_classes == "random":
            if TDC.shape[0] > ASD.shape[0]:  # if no. TDC subjests > no. ASD subjects
                logger.info('Randomly sampling TDC subjects')
                TDC = TDC.iloc[np.random.choice(np.arange(TDC.shape[0]), ASD.shape[0], replace=False)]# randomly sampling from TDC population to make the classes even
            else:
                logger.info('Randomly sampling ASD subjects')
                ASD = ASD.iloc[np.random.choice(np.arange(ASD.shape[0]), TDC.shape[0], replace=False)]# randomly sampling from ASD population to make the classes even
            # logger.info('After random sampling, {} TDCs and {} ASDs with AS>= {} and AS <= {} selected'.format(TDC.shape[0], ASD.shape[0], args.AS[0], args.AS[1]))
            logger.info('After random sampling, {} TDCs and {} ASDs  selected'.format(TDC.shape[0], ASD.shape[0]))
            data = TDC.append(ASD)

#
n_ASD = data[data['control'] == 0].shape[0]
n_TDC = data[data['control'] == 1].shape[0]
logger.info('{} ASD & {} TDC subjects selected'.format(n_ASD, n_TDC))
yy = data['control']
ADOS = data['ADOS']

if args.include_DB:
    logger.info("DB measures: {}   will be included ...".format(args.include_DB))
    if args.include_DB[0] == "all":
        DB = data[["site", "age", "sex", "VIQ", "PIQ", "FIQ"]]
    else:
        DB = data[args.include_DB]
    if args.sex and 'sex' in DB.columns:
        DB.drop('sex', axis=1, inplace=True) # remove sex column if there are same sex subjects
    DB = pd.get_dummies(DB)  # Convert categorical variable site and sex into dummy/indicator variables i.e. one hot encoding
    # integer representation like [0, 1, 2, ..] can not be used directly with scikit-learn estimators, as 
    # these expect continuous input, and would interpret the categories as being ordered, 
    # which is often not desired 
    data = DB.join(data.iloc[:, 9:])
else:
    data = data.iloc[:, 9:]

if args.impute:
    logger.info("Imputaton will be done along columns. strategy = median")
    imputer = Imputer(strategy="median", axis=0)
    data_columns = data.columns
    data_nd_array = imputer.fit_transform(data.values)
    data = pd.DataFrame(data_nd_array, columns=data_columns)

logger.info(data.columns[:10])

if 'FS' in args.data_file:
    intensity_columns = filter(lambda x: "_intensity" in x, data.columns)
    data.drop(intensity_columns, axis=1, inplace=True)  # remove intensity columns
    # is outside the attribute loop to avoid error when both _volume 7 ALL are processed
    eTIV = data['EstimatedTotalIntraCranialVol_volume'] 
    data.drop('EstimatedTotalIntraCranialVol_volume', axis=1, inplace=True)  
elif 'zernike' in args.data_file:
    pass
else:
    pass
   
#
logger.info("{} attribute will be processed.....".format(attributes))
dict_for_attributes = {}
small_dict_for_attributes = {}  # stores only y_test, y_pred, y_pred_prob
for attribute in attributes:
    logger.info('=====================================' + attribute + '==================')
    if '.csv' in args.data_file:
        if (attribute == "_volume" or attribute == "ALL") and (not args.donot_normalize_volume):
            logger.info('Volume features normalized by eTIV')
            volume_columns = filter(lambda x: "_volume" in x, data.columns)
            data[volume_columns] = data[volume_columns].div(eTIV, axis = 'index')
        if attribute == "ALL":
            ALL = "ALL"
            X = data
        else:
            ALL = ""
            attribute_columns = filter(lambda x: re.search(attribute, x), data.columns)
            X = data[attribute_columns]

    features = X.columns      
    # X = X.values       
    y = yy.values
           
    # Remove highly correlated features 
    if args.remove_correlated < 1:
        logger.info("Highly correlatd features with correlation coefficient >= {} will be removed".format(args.remove_correlated))        
        X = remove_highly_correlated(X, threshold=args.remove_correlated)
    logger.info('X shape is {}'.format(X.shape))

    t0 = time()

    # stats for each feature
    if args.stats:
        logger.info('Features stats will be calculated...')
        df_stats = X.apply(get_stats, label=y)
        df_stats.to_csv(stats_filename + '.csv')
        sys.exit()

    if args.n_folds == 1:
        logger.info('No data partition done')
        folds = one_fold(y)
    else:
        small_class_size = min(np.count_nonzero(y == 1), np.count_nonzero(y == 0))
        cv = 5
        if small_class_size < args.n_folds:
            args.n_folds = small_class_size
            args.scoring = "accuracy"
            logger.info("Due to small sample size, n_folds = {} and scoring metric = {}".format(args.n_folds, args.scoring))
        if args.validation == "skf":
            logger.info("{} - fold cross_validation used".format(args.n_folds))
            folds = StratifiedKFold(y, n_folds=args.n_folds, shuffle=True, random_state=np.random.seed(args.seed))
        elif args.validation == "loo":
            logger.info("LeaveOneOut cross_validation used")
            folds = LeaveOneOut(len(y))
        else:
            logger.info("Options are skf, loo")



    #  variables to store result of each fold 
    list_dicts = list() # one dict for each fold and all the folds for one sub cortical structure/attribute are in a list
    small_list_dicts = list() # stores only y_test, y_pred, y_pred_prob
    scores = list()
    AUC = list()
    scores_reg = list()
    Accuracy = list()

    for i, (train_index, test_index) in enumerate(folds):
        if args.fold_no is not None: # process only the given fold no
            if args.fold_no == i:
                logger.info('Processing fold # {}'.format(args.fold_no))
            elif i < args.fold_no:
                continue
            elif i > args.fold_no: # if only one fold run, exit from the loop after that fold
                break

        logger.info("\n Fold # {}".format(i))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # ADOS_train, ADOS_test = ADOS[train_index], ADOS[test_index] # commented because it throws error when SMOTE is done

        # SMOTE
        if args.balance_classes:
            if (args.balance_classes == "SMOTE") or (args.balance_classes == "upsample"):
                n1 = np.count_nonzero(y_train == 1)
                #print(n1)
                n0 = np.count_nonzero(y_train == 0)
                #print(n0)
                if n1 > n0:
                    percent_to_generate = 100*(n1-n0)/n1
                    logger.info("Smaller class is 0 and {} percent of synthetic_samples generated".format(percent_to_generate))
                    big_class = X_train[y_train == 1]
                    small_class = X_train[y_train == 0]
                    ynew = np.concatenate([np.repeat(1, n1), np.repeat(0, n1)])
                    n_large = n1

                else:
                    percent_to_generate = 100*(n0-n1)/n0
                    logger.info("Smaller class is 1 and {} percent of synthetic_samples generated".format(percent_to_generate))
                    big_class = X_train[y_train == 0]
                    small_class = X_train[y_train == 1]
                    ynew = np.concatenate([np.repeat(0, n0), np.repeat(1, n0)])
                    n_large = n0


                print(len(ynew))
                n_synthetic_samples = abs(n1 - n0)

                if args.balance_classes == "SMOTE":
                    logger.info("SMOTE will be done ........")
                    synthetic_samples = SMOTE(T=small_class.values, N=600, k=5)
                    print(synthetic_samples.shape)
                    df_synthetic = pd.DataFrame(synthetic_samples, columns=X_train.columns)
                    X_train = big_class.append(df_synthetic.iloc[:n_large, ])   #.append(df_synthetic)
                elif args.balance_classes == "upsample":
                    logger.info("Random upsampling will be done ........")
                    logger.info('Randomly upsampling small class..')
                    random_samples = small_class.iloc[np.random.choice(np.arange(small_class.shape[0]), abs(n1-n0), replace = True)]
                    X_train = big_class.append(small_class).append(random_samples)
                else:
                    pass


                print(X_train.shape)
                y_train = ynew
   
        ## Feature Selection
        if args.feature_selection:
            logger.info( 'Feature Selecion: X train shape is {}'.format(X_train.shape))
            if args.use_saved_feature_selection:
                if args.leak:
                    saved_feature_selection_file = 'saved_runs/experiment3/LEAK/small_'+args.feature_selection+'____'+args.site+"_"+'.pkl'
                    saved_feature_selection = pd.read_pickle(saved_feature_selection_file)
                    selected_features = saved_feature_selection[attribute][0]['selected_features']
                    # print(saved_feature_selection)
                else:
                    if args.attribute:
                        saved_feature_selection_file = "saved_runs/experiment2/small_"+args.feature_selection+"__"+args.attribute+"___.pkl"
                    elif args.site:
                        saved_feature_selection_file = "saved_runs/experiment3/small_"+args.feature_selection+"____"+args.site+"_.pkl"
                    else:
                        pass
                    logger.info('Selected features used from {}'.format(saved_feature_selection_file))
                    saved_feature_selection = pd.read_pickle(saved_feature_selection_file)

                    selected_features = saved_feature_selection[attribute][i]['selected_features']
                    # selected_features = saved_feature_selection[attribute][i]['features'] # experiment 3 when selected features are used from classification model


                X_train = X_train[selected_features]
                X_test = X_test[selected_features]
                feature_selector = args.feature_selection
            else:
                if args.feature_selection == "randomized_lasso":
                    # logger.info('Selection threshold = {} used'.format(args.selection_threshold))
                    logger.info('Percentile will be used')
                    if args.alpha:
                        alpha = args.alpha
                    else:
                        # Cross validate to get better choice of alpha
                        # Stop the user warnings outputs- they are not necessary for the example
                        # as it is specifically set up to be challenging.
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', UserWarning)
                            warnings.simplefilter('ignore', ConvergenceWarning)
                            lasso_cv = LassoCV(cv=10, alphas=np.logspace(-5, 6, 500), verbose=False).fit(X_train.values, y_train.values) 
                            # lasso_cv = LassoLarsCV(cv=5).fit(X_train.values, y_train.values) # Cross-validated Lasso, using the LARS algorithm

                        logger.info('Chosen alpha from LassoCV: {} '.format(lasso_cv.alpha_))
                        alpha = lasso_cv.alphas_  # randomized lasso will returen all 0 scores if only one aplha is given!!!
                        # print(alpha)
                    feature_selector = RandomizedLasso(alpha=alpha, random_state=0, sample_fraction=0.75, n_resampling=5000,
                                                        verbose=False, n_jobs=-1, selection_threshold=args.selection_threshold)

                elif args.feature_selection == "RFECV_linearSVM":

                    if X_train.shape[0]<25:
                        n_folds = 5  # to avoid error while stratified sampling
                    else:
                        n_folds = 10 
                    feature_selector = RFECV(SVC(kernel="linear"), step=args.step_RFE, cv=StratifiedKFold(y_train,5), scoring="accuracy")
                    # cv-10 prduces error for OLIN

                elif args.feature_selection == "boruta":
                    # class_weight='auto' --> sampling in proportion to y labels
                    forest = RandomForestClassifier(n_jobs=-1, max_depth=7, class_weight='auto')
                     
                    # define Boruta feature selection method
                    feature_selector = boruta.BorutaPy2(forest, n_estimators='auto', perc=args.percentile, max_iter=100, verbose=0)
                else:
                    logger.error('Options are: randomized_lasso, RFECV_linearSVM, boruta')
                    
                # fitting feature selector
                feature_selector.fit(X_train.values, y_train.values)

                logger.info('No. of features selected = {}'.format(np.sum(feature_selector.get_support())))
                selected_features = X_train.columns[feature_selector.get_support()]  # To preserve pandas df since feature names will be handy for feature importances
                if "scores_" in dir(feature_selector):
                    ranked_features = sorted(zip(map(lambda x: round(x, 4), feature_selector.scores_), X_train.columns), reverse=True)
                    logger.info('Top 10 features:{}'.format(ranked_features[:10]))

                if args.feature_selection == "boruta":
                    logger.info('Selected features: {}'.format(X_test.columns[feature_selector.support_]))
                    if len(selected_features) == 0: # if zero features selected, then select tentative features
                        logger.info('!!! 0 optimal features were selected by boruta. So tentative features are used')
                        # selected_features = X_train.columns[feature_selector.support_weak_] # it doesn't work
                        selected_features = X_train.columns[feature_selector.ranking_ == 2]
                        logger.info("Tentative features selected: {}".format(selected_features))

                if args.feature_selection == "randomized_lasso":
                    score_threshold = np.percentile(feature_selector.scores_, 95)
                    selected_features = X_train.columns[feature_selector.scores_ > score_threshold]
                    logger.info('RandomizedLasso: top 5 % features selected')
                    logger.info('Selected features: {}'.format(selected_features))

                # masking with selected_features      
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]
        else:
            feature_selector = None
            selected_features = None

        # Regression
        if args.regression:
            n_features = X.shape[1]
            if args.regression == 'random_forest':
                mtry = np.sqrt(n_features).round()
                param_to_search = {"max_features": np.arange(int(mtry-round(mtry/2)), int(mtry+round(mtry/2)), 1)}
                estimator = RandomForestRegressor(n_estimators=args.n_tree)
            else:
                pass
            regressor = GridSearchCV(estimator=estimator, param_to_search=param_to_search, cv=5, n_jobs=-1)      
            #  model fitting
            regressor.fit(X_train.values, ADOS_train)
            X_train = regressor.transform(X_train)
            X_test = regressor.transform(X_test)

            logger.info("Regression: Best Parameters: {}".format(regressor.best_params_))
            score = regressor.score(X_test, ADOS_test)
            logger.info('Score = {}'.format(score))
            ADOS_pred = regressor.predict(X_test)
            logger.info(ADOS_pred)
            scores_reg.append(score)
        else:
            regressor = None

        # Classification
        if args.classification:
            logger.info( 'Classification: X train shape is {}'.format(X_train.shape))
            if args.classification == "RF":
                n_features = X_train.shape[1]

                if args.search == "grid":
                    #if n_features <= 5:
                        mtry = np.sqrt(n_features).round()
                        param_to_search = {"max_features": np.arange(int(mtry-round(mtry/2)), int(mtry+round(mtry/2)), 1 )}
                elif args.search == "random":
                    param_to_search = {"max_features": sp_randint(1, int(n_features/4)),
                                        "min_samples_split": sp_randint(1, 11),
                                        "min_samples_leaf": sp_randint(1, 11)}
                                        #"n_estimators": sp_randint(1000, 10000)}
                    #param_to_search = {"max_features": sp_randint(1, int(n_features/4))}
                
                else:
                    logger.info('Options are grid and random search')
                class_weight = "auto" if args.balance_classes == "auto" else None
                estimator = RandomForestClassifier(n_estimators=args.n_tree, class_weight=class_weight)
                # estimator = RandomForestClassifier(class_weight=class_weight)
                if args.tree_grid:
                    logger.info("Search on n_estimators will be done.")
                    estimator = RandomForestClassifier(class_weight=class_weight)
                    print(args.tree_grid)
                    param_to_search["n_estimators"] = np.arange(args.tree_grid[0], args.tree_grid[1], args.tree_grid[2])

            elif args.classification == "GBM":
                if args.search == "grid":
                    param_to_search = {"max_depth": range(2, 12), "subsample":[0.5, 0.7]}
                elif args.search == "random":
                    param_to_search = {"max_depth": range(1, 9), "subsample":[0.5, 0.7]}
                # param_to_search = {"max_depth": range(2,12)}

                # param_to_search = {"max_depth": sp_randint(1, 12), "subsample": np.random.uniform(0.5, 0.8, 5)}
                estimator = GradientBoostingClassifier(n_estimators=args.n_tree, learning_rate=0.001)

            elif args.classification == "SVM":
                # param_to_search = {'C': np.logspace(-1, 10, 10), 'gamma': np.logspace(-20, 1, 25)}
                param_to_search = {'C': stats.expon(scale=100), 'gamma': stats.expon(scale=.1)}
                estimator = SVC(kernel="rbf", probability=True)

            elif args.classification == "SVM_linear":
                 param_to_search = {'C': stats.expon(scale=100)}
                 estimator = LinearSVC()

            elif args.classification == "xgboost":
                if args.search == "grid":
                    param_to_search = {"max_depth": range(2, 9), "subsample":[0.5, 0.7]}
                elif args.search == "random":
                    param_to_search = {"max_depth": sp_randint(1, 7), "subsample":np.random.uniform(0.5, 0.8, 5)}
                estimator = xgb.XGBClassifier(n_estimators=args.n_tree, learning_rate=args.learning_rate)

            elif args.classification == 'logistic':
                param_to_search = { 'C' : [10,100, 1000, 10000, 10000]}
                estimator = LogisticRegression()

            elif args.classification == "blending":
                clfs = [RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='gini'),
                RandomForestClassifier(n_estimators=2000, n_jobs=-1, criterion='entropy'),
                RandomForestClassifier(n_estimators=5000, n_jobs=-1, criterion='entropy',max_depth=7),
                RandomForestClassifier(n_estimators=5000, n_jobs=-1, criterion='entropy',min_samples_leaf = 5),
                ExtraTreesClassifier(n_estimators=2000, n_jobs=-1, criterion='gini', max_depth=5),
                ExtraTreesClassifier(n_estimators=5000, n_jobs=-1, criterion='gini'),
                ExtraTreesClassifier(n_estimators=5000, n_jobs=-1, criterion='entropy'),
                ExtraTreesClassifier(n_estimators=2000, n_jobs=-1, criterion='gini', max_depth=7),
                ExtraTreesClassifier(n_estimators=2000, n_jobs=-1, criterion='entropy', max_depth=15),
                ExtraTreesClassifier(n_estimators=2000, n_jobs=-1, criterion='gini', max_depth=20),
                GradientBoostingClassifier(n_estimators=2000, loss='deviance',learning_rate=0.01, subsample=0.5, max_depth=3),
                

                # Creating train and test sets for blending 
                skf1 = list(StratifiedKFold(y_train,  n_folds=5, shuffle=True, random_state = np.random.seed(0)))

                dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
                dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

                for j, clf in enumerate(clfs):
                    print ("model # {}".format(j))
                    dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf1)))
                    for k, (train_index_j, test_index_j) in enumerate(skf1):
                        # print "Blending: Fold", k
                        X_train_k, X_test_k = X_train[train_index_j], X_train[test_index_j]
                        # X_train_k, X_test_k = X_train.iloc[train_index_j], X_train.iloc[test_index_j]
                        y_train_k, y_test_k = y_train[train_index_j], y_train[test_index_j]
                        clf.fit(X_train_k, y_train_k)
                        dataset_blend_train[test_index_j, j] = clf.predict_proba(X_test_k)[:,1]
                        dataset_blend_test_j[:, k] = clf.predict_proba(X_test)[:,1] # Model applied on original test data to give probability
                    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1) # models across test folds are averaged to give probs for original test data

                print
                logger.info("Blending by a logistic classifier")
                param_to_search = {'C': [1, 5, 8, 10, 20, 100]}
                estimator = LogisticRegression()
                X_train = dataset_blend_train
                X_test = dataset_blend_test
            else:
                pass

            if i == 0 or args.fold_no: # print hyperparameter grid for the first fold or when only one fold is run
                logger.info("Parameters to search:{}".format(param_to_search))
            # scoring function
            if args.scoring == "roc_auc_weighted":
                args.scoring = roc_auc_weighted

            # hyperparameter search 
            small_class_size = min(np.count_nonzero(y_train == 1), np.count_nonzero(y_train == 0))
            cv = 5
            if small_class_size < 5:
                cv =  small_class_size
    
            if args.search == "grid":
                if i == 0: # print only for first fold
                    logger.info('Grid search will be done...')
                classifier = GridSearchCV(estimator=estimator, scoring=args.scoring, param_grid=param_to_search, cv=cv, n_jobs=-1)   
            elif args.search == "random":
                if i == 0: # print only for first fold
                    logger.info('Randomized search will be done...')
                if small_class_size <= 2:
                    classifier = RandomizedSearchCV(estimator=estimator, scoring=args.scoring, param_distributions=param_to_search, 
                                                cv=LeaveOneOut(len(y_train)), n_jobs=-1, n_iter=args.n_iter_random_search)
                else:
                    classifier = RandomizedSearchCV(estimator=estimator, scoring=args.scoring, param_distributions=param_to_search, 
                                                cv=cv, n_jobs=-1, n_iter=args.n_iter_random_search)
            else:
                pass


            if (i == 0 and hasattr(classifier.estimator, 'class_weight')):
                logger.info("class_weight = {}".format(estimator.class_weight))
            # model fitting
            classifier.fit(X_train.values, y_train)
            y_pred = classifier.predict(X_test.values)
            y_pred_prob = classifier.predict_proba(X_test.values)
            score = classifier.score(X_test.values,y_test)
            accuracy = accuracy_score(y_test, y_pred) 
            auc = roc_auc_score(y_test, y_pred_prob[:, 1]) # positive class is 1
            features = X_test.columns
            if hasattr(classifier.best_estimator_, 'feature_importances_'):
                feature_importances = classifier.best_estimator_.feature_importances_
            elif hasattr(classifier.best_estimator_, 'coef_'):
                feature_importances = classifier.best_estimator_.coef_
            else:
                feature_importances = None

            logger.info("Classification: Best Parameters: {}".format(classifier.best_params_))
            logger.info('{} = {}, Accuracy = {}, AUC = {}'.format(args.scoring, score, accuracy, auc))
            logger.info('Predicted: {}'.format(y_pred))
            # logger.info('Kappa score = {}'.format(kappa(y_test,y_pred)))
            if args.validation == "skf":
                logger.info(classification_report(y_test, y_pred, target_names=['ASD', 'TDC']))

            scores.append(score)
            AUC.append(auc)
            Accuracy.append(accuracy)
        else:
            classifier = None
            y_pred = None
            y_pred_prob = None
            score = None
            features = None
            feature_importances = None

        # result for one fold
        result_fold = {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred, 'feature_selector': feature_selector, 'regressor': regressor, 'classifier': classifier, 'score': score}
        small_result_fold = {'y_test': y_test, 'y_pred': y_pred, 'y_pred_prob': y_pred_prob, 'features': features, 'feature_importances': feature_importances, 'selected_features': selected_features, 'n_ASD': n_ASD, 'n_TDC': n_TDC}

        logger.info(attribute + ": done in %0.3fs" % (time()-t0))
        list_dicts.append(result_fold)
        small_list_dicts.append(small_result_fold)  # to save small size pickle
        # end of fold

    if (not args.feature_selection) and (not args.classification):
        logger.info("Only two sample t-test done. Exiting.")
        sys.exit()  
    logger.info('Overall mean: {} = {}, Accuracy = {}, AUC = {}'.format(args.scoring, np.mean(scores), np.mean(Accuracy), np.mean(AUC)))
    
    dict_for_attributes[attribute] = list_dicts
    small_dict_for_attributes[attribute] = small_list_dicts
    # end of attribute

pickle.dump(small_dict_for_attributes, open(small_save_file, "wb"))
if args.save_everything:
    pickle.dump(dict_for_attributes, open(save_file, "wb"))
