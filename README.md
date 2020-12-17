Drug Resistance Prediction in Escherichia coli Bacteria Using Machine Learning

Author: 	Mehdi Foroozandeh Shahraki

Abstract

Drug resistance is a growing public health threat with catastrophic medical and economic impacts thus, powerful methodologies are required urgently to tackle this crisis. Recent advancements in high-throughput sequencing technologies as well as the broad arsenal of various machine-learning techniques enable us to approach drug resistance, a traditionally clinical problem, with robust computational tools aiming to understand it and even make phenotypical predictions based on genomic data. In this study, several machine-learning methods have been applied to predict drug-resistance in a familiar cause of a diverse range of infectious diseases Escherichia coli. The results of this project highlight the potential ability of machine-learning in deciphering complex biological problems with significant universal influence.


Below is a list of files in the "./Data and Codes" directory alongside with a short description of what each of them actually do. 

CODES:

1. snp2gene.py:
	
	This module contains several pre-processing functions which are responsible for the following:
		1.1 generate a .csv gene map from .gff genome annotation files
		2.2 convert SNP-based data to gene-based data.

2. BackBone.py:
	
	This module contains:
		2.1 a class of hyper-parameter space for all ML algorithms that are used in this study.
		2.2 a function to read and load .csv gene-based dataset and scale/normalize the loaded data


3. nn_models.py:
	
	This module contains several functions which are responsible for the following:
		3.1 building Dense, CNN, LSTM neural networks from an input of parameters dictionary. 
		3.2 Build LRCN, CNN, feed-forward, and LSTM classifiers
		3.3 masked loss function and masked accuracy function to ignore the isolates with missing labels during the training

4. BO_ML.py:
	
	This module is responsible for the following:
		4.1 performing the nested CV evaluation and Bayesian Hyperparameter optimization for RF, SVM, LR, KNN, GNB, GBC, and XGB algorithms
		4.2 training and saving different machine learning models

5. BO_DL.py:
	
	This module is responsible for the following:
		5.1 performing the nested CV evaluation and Bayesian Hyperparameter optimization for LRCN, CNN, feed-forward, LSTM neural network architectures
		5.2 training and saving different NN-based machine learning models

DATA:

1. Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.48.chromosome.Chromosome.gff3:
		
	This file contains the genome annotation of E coli str_k_12_substr_mg1655

2. gene_based_DS.csv:
	
	This file contains the gene-based dataset

3. Drug_labels.csv:
	
	This file contains the labels for the gene-based dataset


Thank you for reading!