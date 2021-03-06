import warnings, math, sys
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from skopt import gp_minimize, space
import numpy as np
import pandas as pd
from functools import partial
from sklearn import model_selection
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, auc, average_precision_score, precision_recall_curve, accuracy_score

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

import nn_models, BackBone


def NN_multilabel_evaluation_CV(X, Y, drugs, fit_parameters, clf, k):
	kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)

	drugs_res = {}
	for dd in drugs:
		drugs_res[dd] = {
			'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc':[]}
	
	for train_index, test_index in kf.split(X):
		X_train, X_validation = X[train_index], X[test_index]
		y_train, y_validation = Y[train_index], Y[test_index]
		
		es = EarlyStopping(
			monitor='val_loss', min_delta=0.0005, patience=10, mode='min')

		try:
			clf.fit(
				X_train, y_train, validation_data=(X_validation, y_validation),
					**fit_parameters, verbose=0, callbacks=[es])
					
			preds = clf.predict(X_validation)

			Fold_res = {}
			for d_i in range(preds.shape[1]):
				drug_i_results = {
					'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc':[]}
				
				chosen_indices = []
				for i in range(y_validation.shape[0]):
					if y_validation[i, d_i] != -1:
						chosen_indices.append(i)
				
				preds_d = preds[chosen_indices, d_i]
				preds_d = (preds_d > 0.5)
				y_val_d = np.array(y_validation)[chosen_indices, d_i]

				drug_i_results['accuracy'].append(accuracy_score(y_val_d, preds_d))
				drug_i_results['recall'].append(recall_score(y_val_d, preds_d))
				drug_i_results['precision'].append(precision_score(y_val_d, preds_d))
				drug_i_results['f1'].append(f1_score(y_val_d, preds_d))
				
				try:
					drug_i_results['roc_auc'].append(roc_auc_score(y_val_d, preds_d))
				except:
					drug_i_results['roc_auc'].append(0)
				
				try:
					drug_i_results['pr_auc'].append(average_precision_score(y_val_d, preds_d))
				except:
					drug_i_results['pr_auc'].append(0)

				for kk, vv in drug_i_results.items():
					if math.isnan(vv[0]) or np.isinf(vv[0]):
						drug_i_results[kk] = [0]

				Fold_res[drugs[d_i]] = drug_i_results

			for d_c in Fold_res.keys():
				for d_g in Fold_res[d_c].keys():
					drugs_res[d_c][d_g].append(Fold_res[d_c][d_g][0])
		except:
			print('an important fold exception was made')
			continue 
			
	for ab in drugs_res.keys():
		for met in drugs_res[ab].keys():
			drugs_res[ab][met] = np.mean(drugs_res[ab][met]) 

	drugs_res = pd.DataFrame(drugs_res)
	KFCV_res = drugs_res.mean(axis=1)

	return drugs_res, KFCV_res


def optimize_object(params, param_names, drugs, model, X, Y, k=10, target_metric='roc_auc'):
	params = dict(zip(param_names, params))

	fit_parameters = {
		'batch_size': params['batch_size'], 
		'epochs': params['epochs']
		}

	for fp in fit_parameters.keys():
		del params[fp]

	if model == 'lrcn':
		clf = nn_models.build_lrcn_clf(params)

	elif model == 'cnn':
		clf = nn_models.build_cnn_clf(params)

	elif model == 'lstm':
		clf = nn_models.build_lstm_clf(params)

	elif model == 'dense':
		clf = nn_models.build_dense_clf(params)

	drugs_res, KFCV_res = NN_multilabel_evaluation_CV(X, Y, drugs, fit_parameters, clf, k)
	
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
		n_calls=100,
		acq_func='EI')

	return dict(zip(param_names, results.x)), results.fun

#    ======================================================================================================     #

if __name__ == '__main__':
	model_name = sys.argv[1]
	nns = ['lrcn', 'cnn', 'lstm', 'dense']

	if model_name not in nns:
		exit()
		
	X_g, Y_g = BackBone.load_gene_ds('gene_based_DS.csv', 'Drug_labels.csv', scale_strategy='gene_length')
	X_g = np.reshape(np.array(X_g).astype(np.float32), [X_g.shape[0], X_g.shape[1], 1])
	ABs = Y_g.columns
	Y_g = np.array(Y_g).astype(np.float32)
	HPS = BackBone.hyperparameter_space()

	best_val_roc_auc = []
	drugs_res = {}

	for dd in ABs:
		drugs_res[dd] = {
			'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc':[]}

	kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
	for train_index, test_index in kf.split(X_g):
		X_train, X_test = X_g[train_index], X_g[test_index]
		y_train, y_test = Y_g[train_index], Y_g[test_index]

		try:
			if model_name == 'lrcn':
				best_params, best_val_score = run_BO('lrcn', HPS.LRCN_space, ABs, HPS.LRCN_names, X_train, y_train, cv_k=9)
			
			elif model_name == 'cnn':
				best_params, best_val_score = run_BO('cnn', HPS.CNN_space, ABs, HPS.CNN_names, X_train, y_train, cv_k=9)
			
			elif model_name == 'lstm':
				best_params, best_val_score = run_BO('lstm', HPS.LSTM_space, ABs, HPS.LSTM_names, X_train, y_train, cv_k=9)

			elif model_name == 'dense':
				best_params, best_val_score = run_BO('dense', HPS.Dense_space, ABs, HPS.Dense_names, X_train, y_train, cv_k=9)

			best_val_roc_auc.append(-1 * float(best_val_score))

			if float(-1 * best_val_score) >= max(best_val_roc_auc):
				with open('models/final/Saved_{}_params.txt'.format(model_name), 'w') as f:
					f.write(str(best_params))

			fit_parameters = {
				'batch_size': best_params['batch_size'], 
				'epochs': best_params['epochs']
				}

			for fp in fit_parameters.keys():
				del best_params[fp]

			if model_name == 'lrcn':
				clf = nn_models.build_lrcn_clf(best_params)
			
			elif model_name == 'cnn':
				clf = nn_models.build_cnn_clf(best_params)
			
			elif model_name == 'lstm':
				clf = nn_models.build_lstm_clf(best_params)

			elif model_name == 'dense':
				clf = nn_models.build_dense_clf(best_params)
			
			clf.fit(X_train, y_train, **fit_parameters, verbose=0)
			preds = clf.predict(X_test)

			Fold_res = {}
			for d_i in range(preds.shape[1]):
				drug_i_results = {
					'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'pr_auc':[]}
				
				chosen_indices = []
				for i in range(y_test.shape[0]):
					if y_test[i, d_i] != -1:
						chosen_indices.append(i)
				
				preds_d = preds[chosen_indices, d_i]
				preds_d = (preds_d > 0.5)
				y_val_d = np.array(y_test)[chosen_indices, d_i]

				drug_i_results['accuracy'].append(accuracy_score(y_val_d, preds_d))
				drug_i_results['recall'].append(recall_score(y_val_d, preds_d))
				drug_i_results['precision'].append(precision_score(y_val_d, preds_d))
				drug_i_results['f1'].append(f1_score(y_val_d, preds_d))

				try:
					drug_i_results['roc_auc'].append(roc_auc_score(y_val_d, preds_d))
				except:
					drug_i_results['roc_auc'].append(0)
				
				try:
					drug_i_results['pr_auc'].append(average_precision_score(y_val_d, preds_d))
				except:
					drug_i_results['pr_auc'].append(0)
				
				for kk, vv in drug_i_results.items():
					if math.isnan(vv[0]) or np.isinf(vv[0]):
						drug_i_results[kk] = [0]

				Fold_res[ABs[d_i]] = drug_i_results

			if float(-1 * best_val_score) >= max(best_val_roc_auc):
				clf.save('models/final/Saved_{}_model.h5'.format(model_name))
				plot_model(clf, to_file='models/final/Saved_{}_model.png'.format(model_name))
				saved_model_test_results = pd.DataFrame(Fold_res)
				saved_model_test_results.to_csv('models/final/Saved_{}_Test_results.csv'.format(model_name))

		except:
			print('skipped a fold')
			continue

		for d_c in Fold_res.keys():
			for d_g in Fold_res[d_c].keys():
				drugs_res[d_c][d_g].append(Fold_res[d_c][d_g][0])
	
	for ab in drugs_res.keys():
		for met in drugs_res[ab].keys():
			drugs_res[ab][met] = [np.mean(drugs_res[ab][met]), np.std(drugs_res[ab][met])]

	drug_res = pd.DataFrame(drugs_res)
	print('{}_Overall_res\n'.format(model_name), drug_res)
	drug_res.to_csv('models/final/overall_{}_Test_results.csv'.format(model_name))