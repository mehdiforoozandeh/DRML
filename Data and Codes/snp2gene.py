import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from functools import partial
import csv
import os, sys

def read_loc_from_gff(file_name): #makes the map
    
    with open(file_name, 'r') as f:
        l1 = f.readlines()
    l2 = []
    
    for i in l1:
        l2.append(i.split('\t'))
    
    l3 = []
    for i in l2:
        if len(i)>=2:
            if i[2] == 'gene':
                l3.append(i)

    df = pd.DataFrame(l3, columns=['_', '_', '_', 'start_loc', 'end_loc', '_', '_', '_', 'gene_info'])

    return df


def clean_snps(in_file, out_file):
    with open(in_file, 'r') as t:
        csv = t.readlines()
        print(len(csv))
        print(len(csv[2].split(',')))
        a = csv[0].split(',')
        csv = csv[1:]
        print(len(csv))

        b = []
        temp = ''
        for i in range(len(a)):
            if a[i].count('"') == 2:
                b.append(a[i])
            else:
                temp += a[i]
                if temp.count('"') == 2:
                    b.append(temp)
                    temp = ''

        for i in range(len(b)):
            while '"' in b[i]:
                b[i] = b[i].replace('"', '')
            while ' ' in b[i]:
                b[i] = b[i].replace(' ', '_')

        with open(out_file, 'w') as t3:
            t3.write(','.join(b))
            for i in range(len(csv)):
                t3.write(','.join(csv[i].split(',')))

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


def processing_node(snps, the_map, genes_column_names):
    gene_based = pd.DataFrame(np.zeros((snps.shape[0], the_map.shape[0]), dtype=int), columns=genes_column_names)

    track = 0
    for (columnName0, columnData0) in snps.iteritems():

        track += 1
        if track % 50000 == 0:
            print ("{} out of {}.".format(track, snps.shape[1]))
        
        for (columnName1, columnData1) in gene_based.iteritems():
            
            if int(columnName1.split('_')[1]) <= int(columnName0.split('_')[0]) < int(columnName1.split('_')[2]):

                for i in range(gene_based.shape[0]):
                    
                    if int(columnData0[i]) == 1:
                        gene_based[columnName1][i] += len(columnName0.split('_')[1])

                break
    
    return gene_based

def snp_to_gene(snp_file, map_file, names_of_ABs, n_threads=1, out_gene_file='gene_based_DS.csv', out_label_file='Drug_labels.csv'):
    the_map = pd.read_csv(map_file)
    the_map = the_map.drop(['Unnamed: 0'], axis=1)
    print('map shape:    ', the_map.shape)

    genes_column_names = []
    for i in range(the_map.shape[0]):
        genes_column_names.append(str(the_map['gene_info'][i].split(';')[1].replace('Name=', '')) +
                                 '_' + str(the_map['start_loc'][i]) + 
                                 '_' + str(the_map['end_loc'][i]))

    snps, anti_bios = load_snp_DS(snp_file, names_of_ABs)

    if n_threads == 1:
        gene_based = processing_node(snps, the_map, genes_column_names)
    
    elif n_threads > 1:
        pool = mp.Pool(n_threads)
        
        print('breakin down snps into {} files'.format(str(n_threads)))

        temp_files = []
        for i in range(0, snps.shape[0], int(snps.shape[0]/n_threads)):
            tsnp = snps.loc[i: i+int(snps.shape[0]/n_threads)-1, :]
            tsnp = tsnp.reset_index()
            tsnp = tsnp.drop(['index'], axis=1)
            temp_files.append(tsnp)
        del snps
        
        print('Starting to Multiprocess ...')
        MP_Results = pool.map(partial(processing_node, the_map=the_map, genes_column_names=genes_column_names), temp_files)
        gene_based = pd.concat(MP_Results)

    print(gene_based.shape)

    anti_bios.to_csv(out_label_file)
    gene_based.to_csv(out_gene_file)
    print('DS successfully Generated')


if __name__ == '__main__':
    # res = read_loc_from_gff('Escherichia coli str. K-12 substr. MG1655/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.46.gff3')
    # res.to_csv("ecoli_k12_MG1655_genemap.csv")

    # with open('complete_DS.csv', 'r') as f:
    #   a = f.readlines()
    #   header = a[0]
    #   a = a[1:]
    #   breaking = int(len(a)/50)
    #   j = 0
    #   v = open('CompleteDS/0_DS.csv', 'w')
    #   v.write(header)
    #   k = 0
    #   for i in range(len(a)):
    #       j += 1
    #       v.write(a[i])
    #       if j% breaking == 0: 
    #           k += 1
    #           v = open('CompleteDS/' + str(k) + '_DS.csv', 'w')
    #           v.write(header)

    ABs= ['amikacin', 'amoxicillin', 'ampicillin', 'aztreonam', 'cefalotin', 'cefepime', 
                    'cefotaxime', 'cefoxitin', 'ceftazidime', 'ceftriaxone', 'cefuroxime', 'ciprofloxacin', 
                    'ertapenem', 'gentamicin', 'imipenem', 'meropenem', 'tetracycline']
                    
    start = datetime.now()

    snp_to_gene('complete_DS.csv', 'ecoli_k12_MG1655_genemap.csv', ABs, n_threads=10, 
            out_gene_file= 'output_files/gene_based_DS.csv', out_label_file='output_files/Drug_labels.csv')

    end = datetime.now() 
    print("The whole thing took {} ".format(end-start))

    
