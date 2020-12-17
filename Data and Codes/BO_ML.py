import warnings, math, sys, joblib
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from skopt import gp_minimize, space
import numpy as np
import pandas as pd
from functools import partial
from sklearn import model_selection
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, auc, average_precision_score, precision_recall_curve, accuracy_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

import BackBone

def optimize_object(params, param_names, drugs, model, X, Y, k=10, target_metric='roc_auc'):
	params = dict(zip(param_names, params))

	clf = model(**params)
	kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=73)

	drugs_res = {}
	for dd in drugs:
		drugs_res[dd] = None

	for d_i in range(Y.shape[1]):
		chosen_indices = []

		for i in range(Y.shape[0]):
			if Y[i, d_i] != -1:
				chosen_indices.append(i)

		X_d = np.array(X[chosen_indices, :])
		Y_d = np.array(Y[chosen_indices, d_i])

		drug_i_results = {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc':[]}

		for train_index, test_index in kf.split(X_d):
			X_train, X_validation = X_d[train_index], X_d[test_index]
			y_train, y_validation = Y_d[train_index], Y_d[test_index]
			
			try:
				clf.fit(X_train, y_train)
				preds = clf.predict(X_validation)

				drug_i_results['accuracy'].append(accuracy_score(y_validation, preds))
				drug_i_results['recall'].append(recall_score(y_validation, preds))
				drug_i_results['precision'].append(precision_score(y_validation, preds))
				drug_i_results['f1'].append(f1_score(y_validation, preds))
				
				try:
					drug_i_results['roc_auc'].append(roc_auc_score(y_validation, preds))
				except:
					drug_i_results['roc_auc'].append(0)
				
				try:
					drug_i_results['pr_auc'].append(average_precision_score(y_validation, preds))
				except:
					drug_i_results['pr_auc'].append(0)

			except:
				print('an important fold exception was made')
				continue 

		for kee in drug_i_results.keys():
			drug_i_results[kee] = np.mean(drug_i_results[kee])
		
		drugs_res[drugs[d_i]] = drug_i_results
	
	drugs_res = pd.DataFrame(drugs_res)
	KFCV_res = drugs_res.mean(axis=1)
	print(drugs_res)

	return -1 * np.mean(KFCV_res[target_metric])


def run_BO(clf, param_space, drugs, param_names, X, Y, cv_k=10):
	optimization_function = partial(
		optimize_object,
		param_names = param_names,
		model=clf,
		X=X,
		Y=Y,
		drugs=drugs,
		k=cv_k
	)

	results = gp_minimize(
		optimization_function,
		dimensions= param_space,
		n_calls=60,
		acq_func='EI')

	return dict(zip(param_names, results.x)), results.fun

#    ======================================================================================================     #

if __name__ == '__main__':

	model_name = sys.argv[1]
	algs = {'rf':RandomForestClassifier, 'svm':SVC, 'lr':LogisticRegression, 
		'knn':KNeighborsClassifier, 'gbc':GradientBoostingClassifier, 
		'gnb':GaussianNB, 'xgb':XGBClassifier}

	if model_name not in algs:
		exit()

	X_g, Y_g = BackBone.load_gene_ds('gene_based_DS.csv', 'Drug_labels.csv', scale_strategy='gene_length')
	X_g = np.array(X_g).astype(np.float32)
	ABs = Y_g.columns
	Y_g = np.array(Y_g)
	print(X_g.shape)
	print(Y_g.shape)

	HPS = BackBone.hyperparameter_space()

	model_lib = {
		RandomForestClassifier:[HPS.RF_space, HPS.RF_names], 
		SVC:[HPS.SVM_space, HPS.SVM_names], 
		LogisticRegression:[HPS.LR_space, HPS.LR_names], 
		KNeighborsClassifier:[HPS.KNN_space, HPS.KNN_names], 
		GradientBoostingClassifier:[HPS.GBclf_space, HPS.GBclf_names], 
		GaussianNB:[HPS.GNB_space, HPS.GNB_names],
		XGBClassifier:[HPS.XGB_space, HPS.XGB_names]
	}

	MODEL = algs[model_name]
	HP = model_lib[MODEL]

	best_val_roc_auc = []

	drugs_res = {}
	for dd in ABs:
		drugs_res[dd] = {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc':[]}


	kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=73)
	for train_index, test_index in kf.split(X_g):
		X_train, X_test = X_g[train_index], X_g[test_index]
		y_train, y_test = Y_g[train_index], Y_g[test_index]

		try:
			best_params, best_val_score = run_BO(MODEL, HP[0], ABs, HP[1], X_train, y_train, cv_k=9)

			best_val_roc_auc.append(-1 * float(best_val_score))

			if float(-1 * best_val_score) >= max(best_val_roc_auc):
				with open('models/final/Saved_{}_params.txt'.format(MODEL), 'w') as f:
					f.write(str(best_params))

			clf = MODEL(**best_params)

			for d_i in range(y_train.shape[1]):

				chosen_indices = []
				for i in range(y_train.shape[0]):

					if y_train[i, d_i] != -1:
						chosen_indices.append(i)

				X_train_d = np.array(X_train[chosen_indices, :])
				y_train_d = np.array(y_train[chosen_indices, d_i])

				chosen_indices = []
				for i in range(y_test.shape[0]):

					if y_test[i, d_i] != -1:
						chosen_indices.append(i)

				X_test_d = np.array(X_test[chosen_indices, :])
				y_test_d = np.array(y_test[chosen_indices, d_i])

				clf.fit(X_train_d, y_train_d)
				preds = clf.predict(X_test_d)

				drugs_res[ABs[d_i]]['accuracy'].append(accuracy_score(y_test_d, preds))
				drugs_res[ABs[d_i]]['recall'].append(recall_score(y_test_d, preds))
				drugs_res[ABs[d_i]]['precision'].append(precision_score(y_test_d, preds))
				drugs_res[ABs[d_i]]['f1'].append(f1_score(y_test_d, preds))

				try:
					drugs_res[ABs[d_i]]['roc_auc'].append(roc_auc_score(y_test_d, preds))
				except:
					drugs_res[ABs[d_i]]['roc_auc'].append(0)
				
				try:
					drugs_res[ABs[d_i]]['pr_auc'].append(average_precision_score(y_test_d, preds))
				except:
					drugs_res[ABs[d_i]]['pr_auc'].append(0)

			if float(-1 * best_val_score) >= max(best_val_roc_auc):
				joblib.dump(clf, 'models/final/Saved_{}_model.sav'.format(MODEL))

		except:
			print('skipped a fold')
			continue
	
	for dr in drugs_res.keys():
		for met in drugs_res[dr].keys():
			drugs_res[dr][met] = [np.mean(drugs_res[dr][met]), np.std(drugs_res[dr][met])]

	drug_res = pd.DataFrame(drugs_res)
	print('{}_Overall_res\n'.format(MODEL), drug_res)
	drug_res.to_csv('models/final/overall_{}_Test_results.csv'.format(MODEL))