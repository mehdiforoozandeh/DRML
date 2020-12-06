import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.naive_bayes import GaussianNB



def load_snp_DS(filename, ABs, chunksize=100, verbose=True):
	df_chunk = pd.read_csv(filename, chunksize=chunksize)
	snps = []
	anti_bios = []

	j = 0
	for chunk in df_chunk:

		j += 1
		if verbose == True and j % 10 == 0:
			print('loaded {} rows'.format(str(chunksize*j)))

		chunk = chunk.drop(['Unnamed: 0', 'id', 'V1'], axis=1)
		chunk_ABs = chunk[ABs]
		
		chunk = chunk.drop(ABs, axis=1)
		chunk = chunk.astype('int8')
		
		anti_bios.append(chunk_ABs)
		snps.append(chunk)

	snps = pd.concat(snps)
	anti_bios = pd.concat(anti_bios)

	return snps, anti_bios

def load_gene_ds(gene_file, label_file):
	gene_ds = pd.read_csv(gene_file)
	gene_ds = gene_ds.drop(['Unnamed: 0'], axis=1)

	label_ds = pd.read_csv(label_file)
	label_ds = label_ds.drop(['Unnamed: 0'], axis=1)

	return gene_ds, label_ds


def return_param(model='all'):

	models_parameters = {
		RandomForestClassifier:{
			'n_estimators': [10, 50, 100, 500, 1000],
			'criterion': ['gini','entropy'],
			'min_samples_split': [2,3]},

		SVC:{
			'kernel': ['rbf', 'linear', 'poly'], 
			'C':[1, 2, 5, 10, 20, 50],
			'decision_function_shape':['ovo', 'ovr'],
			'gamma':['scale', 'auto']},

		LogisticRegression:{
			'solver':['liblinear', 'lbfgs', 'newton-cg'],
			'C':[1.0, 0.9, 0.8]},

		KNeighborsClassifier:{
			'n_neighbors': [3, 4, 5, 6]},

		AdaBoostClassifier:{
			'n_estimators':[50, 100],
			'algorithm': ['SAMME', 'SAMME.R']},
		
		GradientBoostingClassifier:{
			'loss':['deviance', 'exponential']},
		
		GaussianNB:{
			'var_smoothing': [1e-9, 1e-8]}
			}

	if model in models_parameters.keys():
		return models_parameters[model]

	elif model == 'all':
		return models_parameters

def smote(x, y, k=3):
	smote = SMOTE(k_neighbors=k)
	x_resampled, y_resampled = smote.fit_resample(x, y) 
	return x_resampled, y_resampled

def dimen_reduc(score_func, X, Y, strat='FDR',  K=10, based_on_pval=True):  # score_func : either f_classif or Chi2

	if strat == 'KBest':
		selector = SelectKBest(score_func, k=K)
		selector.fit(X, Y)

		if based_on_pval == True:
			pvals = selector.pvalues_
			K = 0
			for i in pvals:
				if i <= 0.05:
					K += 1

			X_new = SelectKBest(score_func, k=K).fit_transform(X, Y)

		return X_new, Y

	elif strat == 'FDR':
		X_new = SelectFdr(score_func, alpha=0.05).fit_transform(X, Y)
		return X_new, Y

	elif strat == 'FPR':
		X_new = SelectFpr(score_func, alpha=0.05).fit_transform(X, Y)
		return X_new, Y

def hpo(model, params, X_val, Y_val, folds=5, mp=True, strat='grid'):
	if mp == True:
		n_threads = folds
	else:
		n_threads = 1

	if strat == 'grid':
		clf = model()
		clf = GridSearchCV(
			estimator=clf, param_grid=params, 
			cv=folds, n_jobs=n_threads, scoring='roc_auc')
			
		clf.fit(X_val, Y_val)
		return clf.best_params_ , clf.best_score_
	
	elif strat == 'BO':
		pass

def evaluate(clf, X_data, Y_data, strat='KFCV',  k=5, SMOTE = False, split_size=0.176):

	if strat == 'KFCV':
		KFCV_res = {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc':[]}
		kf = StratifiedKFold(n_splits=k, random_state=42)

		for train_index, test_index in kf.split(X_data, Y_data):
			X_train, X_test = X_data[train_index], X_data[test_index]
			y_train, y_test = Y_data[train_index], Y_data[test_index]

			if SMOTE == True:
				X_train, y_train = smote(X_train, y_train)

			clf.fit(X_train, y_train)
			preds = clf.predict(X_test)
			KFCV_res['accuracy'].append(accuracy_score(y_test, preds))
			KFCV_res['recall'].append(recall_score(y_test, preds))
			KFCV_res['precision'].append(precision_score(y_test, preds))
			KFCV_res['f1'].append(f1_score(y_test, preds))

			try:
				fpr, tpr, threshold = roc_curve(y_test, preds)
				roc_auc = auc(fpr, tpr)
				KFCV_res['roc_auc'].append(roc_auc)
			except:
				KFCV_res['roc_auc'].append(0)
			
			try:
				precision, recall, thresholds = precision_recall_curve(y_test, preds)
				pr_auc = auc(precision, recall)
				KFCV_res['pr_auc'].append(pr_auc)
			except:
				KFCV_res['pr_auc'].append(0)

		for kee in KFCV_res.keys():
			KFCV_res[kee] = np.mean(KFCV_res[kee])

		return KFCV_res
	
	elif strat == 'tt_split':
		tt_res = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f1': 0, 'roc_auc': 0, 'pr_auc': 0}
		X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=split_size, random_state=42)

		if SMOTE == True:
			X_train, y_train = smote(X_train, y_train)

		clf.fit(X_train, y_train)
		preds = clf.predict(X_test)

		tt_res['accuracy'] = accuracy_score(y_test, preds)
		tt_res['recall'] = recall_score(y_test, preds)
		tt_res['precision'] = precision_score(y_test, preds)
		tt_res['f1'] = f1_score(y_test, preds)

		try:
			fpr, tpr, threshold = roc_curve(y_test, preds)
			roc_auc = auc(fpr, tpr)
			tt_res['roc_auc'] = roc_auc
		except:
			tt_res['roc_auc'] = 0

		try:	
			precision, recall, thresholds = precision_recall_curve(y_test, preds)
			pr_auc = auc(precision, recall)
			tt_res['pr_auc'] = pr_auc
		except:
			tt_res['pr_auc'] = 0
			
		return tt_res

def explore_models(data, labels, output_filename,  k_out=5, feat_select=False, feat_select_strat='FDR', feat_select_scoreFunc=chi2):
	model_parameters = return_param()
	
	exploration_results = []
	for model in model_parameters.keys():
		
		print(str(model))
		for drug in labels.columns:
			chosen_indices = []

			for i in range(labels.shape[0]):

				if labels[drug][i] != -1:
					chosen_indices.append(i)

			X = np.array(data.iloc[chosen_indices])
			Y = np.array(labels[drug][chosen_indices])

			balance = 0
			for j in Y:
				if j == 1:
					balance += 1
			balance = (float(balance)/len(Y), 1-(float(balance)/len(Y)))

			# X_train_test, X_validation, y_train_test, y_validation = train_test_split(X, Y, test_size=0.15, random_state=42)

			# nested cross-validation
			nested_CV_res = {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc':[]}
			kf = StratifiedKFold(n_splits=k_out, random_state=42)

			for train_index, test_index in kf.split(X, Y):
				X_train_test, X_validation = X[train_index], X[test_index]
				y_train_test, y_validation = Y[train_index], Y[test_index]

				best_params, best_score = hpo(model, model_parameters[model], X_validation, y_validation, folds=5, mp=True)
				clf = model(**best_params)
				eval_results = evaluate(clf, X_train_test, y_train_test, strat='KFCV',  k=5, SMOTE=True)

				for key_ in eval_results.keys():
					nested_CV_res[key_].append(eval_results[key_])
			
			for kee in nested_CV_res.keys():
				nested_CV_res[kee] = np.mean(nested_CV_res[kee])
			
			exploration_results.append({'model': str(model), 'drug':str(drug), 'data_shape':str(X.shape), 'balance': balance, **eval_results})
	
	exploration_results = pd.DataFrame(exploration_results)
	exploration_results.to_csv(output_filename)

if __name__ == "__main__":

	X_g, Y_g = load_gene_ds('E.coli_DS/output_files/gene_based_DS.csv', 'E.coli_DS/output_files/Drug_labels.csv')
	explore_models(X_g, Y_g, output_filename='model_exploration_results_EColi_genes.csv')

	# ABs= ['amikacin', 'amoxicillin', 'ampicillin', 'aztreonam', 'cefalotin', 'cefepime', 
    #                 'cefotaxime', 'cefoxitin', 'ceftazidime', 'ceftriaxone', 'cefuroxime', 'ciprofloxacin', 
    #                 'ertapenem', 'gentamicin', 'imipenem', 'meropenem', 'tetracycline']

	# X_s, Y_s = load_snp_DS('E.coli_DS/output_files/complete_DS.csv', ABs, chunksize=100, verbose=True)
	# explore_models(X_s, Y_s, output_filename='model_exploration_results_EColi_snps.csv')