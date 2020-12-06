from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from model_comparison import *




import numpy as np

def masked_loss_function(y_true, y_pred):
	mask = K.cast(K.not_equal(y_true, -1), K.floatx())
	return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
	dtype = K.floatx()
	total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
	correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
	return correct / total


def build_cnn(
	model, archit_list, filter_size_list, strides, pooling_strategy, pooling_kernel_size, dropout):

	for l in range(len(archit_list)):
		model.add(layers.Conv1D(archit_list[l], filter_size_list[l], 
			strides=strides, padding='same', activation='relu'))

		if pooling_strategy == 'max':
			model.add(layers.MaxPooling1D(pool_size=pooling_kernel_size, padding='same'))
		elif pooling_strategy == 'avg':
			model.add(layers.AveragePooling1D(pool_size=pooling_kernel_size, padding='same'))

		if l%2 == 1:
			model.add(layers.Dropout(dropout))
	
	return model


def build_lstm(model, archit_list, dropout_rate=0.0):
	for l in range(len(archit_list)):

		if l == 0 and l != int(len(archit_list)-1):
			model.add(layers.Bidirectional(layers.LSTM(archit_list[l], dropout=dropout_rate, return_sequences=True)))
		
		elif l == 0 and l == int(len(archit_list)-1):
			model.add(layers.Bidirectional(layers.LSTM(archit_list[l], dropout=dropout_rate, return_sequences=False)))    

		elif l != 0 and l != int(len(archit_list)-1):
			model.add(layers.Bidirectional(layers.LSTM(archit_list[l], dropout=dropout_rate, return_sequences=True)))

		elif l != 0 and l == int(len(archit_list)-1):
			model.add(layers.Bidirectional(layers.LSTM(archit_list[l], dropout=dropout_rate, return_sequences=False)))

	return model

def build_dense(model, archit_list, dropout_rate=0):
	for l in range(len(archit_list)):
		model.add(layers.Dense(archit_list[l], activation='relu'))
		if dropout_rate != 0 and l != int(len(archit_list)-1):
			model.add(layers.Dropout(dropout_rate))
	model.add(layers.Dense(17, activation='sigmoid'))
	return model

def build_lrcn_clf(params):

	params_to_modify = ['cnn_n_layers', 'cnn_n_units_per_layer', 'cnn_filter_size', 'lstm_n_layers',
        'lstm_n_units_per_layer', 'dense_n_layers', 'dense_n_units_per_layer']

	params['cnn_archit'] = [
		int(params['cnn_n_units_per_layer']) for _ in range(params['cnn_n_layers'])]

	params['cnn_filter_size_list'] = [
		int(_ * params['cnn_filter_size']) + 1  for _ in range(params['cnn_n_layers'])]
	
	params['lstm_archit'] = [
		int(params['lstm_n_units_per_layer']) for _ in range(params['lstm_n_layers'])]
	
	params['dense_archit'] = [
		int(params['dense_n_units_per_layer']) for _ in range(params['dense_n_layers'])]

	params['cnn_strides'] = int(params['cnn_strides'])
	params['cnn_pooling_kernel_size'] = int(params['cnn_pooling_kernel_size'])
	for p2m in params_to_modify:
		del params[p2m]

	model = Sequential()
	model = build_cnn(
		model, params['cnn_archit'], params['cnn_filter_size_list'], 
		params['cnn_strides'], params['cnn_pooling_strat'], params['cnn_pooling_kernel_size'], params['cnn_DO'])
	
	model = build_lstm(model, params['lstm_archit'], params['lstm_DO'])

	model = build_dense(model, params['dense_archit'], params['dense_DO'])

	model.compile(
		loss= masked_loss_function, optimizer=params['optim'](params['lr']), 
		metrics=[masked_accuracy])
	model.build((None, 4140, 1))
	model.summary()
	plot_model(model, to_file='models/LRCN_model_summary.png')

	return model

def build_cnn_clf(params):
	params_to_modify = [
		'cnn_n_layers', 'cnn_n_units_per_layer', 'cnn_filter_size', 
		'dense_n_layers', 'dense_n_units_per_layer']

	params['cnn_archit'] = [
		int(params['cnn_n_units_per_layer']) for _ in range(params['cnn_n_layers'])]

	params['cnn_filter_size_list'] = [
		int(_ * params['cnn_filter_size']) + 1  for _ in range(params['cnn_n_layers'])]
	
	params['dense_archit'] = [
		int(params['dense_n_units_per_layer']) for _ in range(params['dense_n_layers'])]

	params['cnn_strides'] = int(params['cnn_strides'])
	params['cnn_pooling_kernel_size'] = int(params['cnn_pooling_kernel_size'])
	for p2m in params_to_modify:
		del params[p2m]

	model = Sequential()
	model = build_cnn(
		model, params['cnn_archit'], params['cnn_filter_size_list'], 
		params['cnn_strides'], params['cnn_pooling_strat'], params['cnn_pooling_kernel_size'], params['cnn_DO'])
	
	model.add(layers.Flatten())
	model = build_dense(model, params['dense_archit'], params['dense_DO'])

	model.compile(
		loss= masked_loss_function, optimizer=params['optim'](params['lr']), 
		metrics=[masked_accuracy])
	model.build((None, 4140, 1))
	model.summary()
	plot_model(model, to_file='models/CNN_model_summary.png')

	return model

def build_lstm_clf(params):
	params_to_modify = [
		'lstm_n_layers', 'lstm_n_units_per_layer', 
		'dense_n_layers', 'dense_n_units_per_layer']

	params['lstm_archit'] = [
		int(params['lstm_n_units_per_layer']) for _ in range(params['lstm_n_layers'])]
	
	params['dense_archit'] = [
		int(params['dense_n_units_per_layer']) for _ in range(params['dense_n_layers'])]

	for p2m in params_to_modify:
		del params[p2m]

	model = Sequential()
		
	model = build_lstm(model, params['lstm_archit'], params['lstm_DO'])

	model = build_dense(model, params['dense_archit'], params['dense_DO'])

	model.compile(
		loss= masked_loss_function, optimizer=params['optim'](params['lr']), 
		metrics=[masked_accuracy])
	model.build((None, 4140, 1))
	model.summary()
	plot_model(model, to_file='models/LSTM_model_summary.png')

	return model

def build_dense_clf(params):
	params_to_modify = ['dense_n_layers', 'dense_n_units_per_layer']
	
	params['dense_archit'] = [
		int(params['dense_n_units_per_layer']) for _ in range(params['dense_n_layers'])]

	for p2m in params_to_modify:
		del params[p2m]

	model = Sequential()
	model.add(layers.Flatten())
	model = build_dense(model, params['dense_archit'], params['dense_DO'])

	model.compile(
		loss= masked_loss_function, optimizer=params['optim'](params['lr']), 
		metrics=[masked_accuracy])

	model.build((None, 4140, 1))
	model.summary()
	plot_model(model, to_file='models/Dense_model_summary.png')

	return model

if __name__ == '__main__':
	params = {
		'cnn_archit': [256],
		'cnn_filter_size_list': [1],
		'cnn_strides':2,
		'pooling_strat': 'avg',
		'pooling_kernel_size':2,
		'cnn_DO': 0.1,
		'lstm_archit':[32, 16, 8],
		'lstm_DO':0.1,
		'dense_archit':[64, 32, 16],
		'dense_DO': 0.1,
		'lr': 0.001,
		'optim': Adam
	}

	X_g, Y_g = load_gene_ds('E.coli_DS/output_files/gene_based_DS.csv', 'E.coli_DS/output_files/Drug_labels.csv')
	X_g = np.reshape(np.array(X_g).astype(np.float32), [X_g.shape[0], X_g.shape[1], 1])
	Y_g = np.array(Y_g).astype(np.float32)
	print(X_g.shape)
	print(Y_g.shape)

	# exit()
	model = build_lrcn_clf(params)
	model.fit(
		x=X_g,y=Y_g, batch_size=64, epochs=12, 
		validation_split=0.3)