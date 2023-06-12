
"""
rotinas de cálculo de métrica
"""
import sys
import os
import time
import ast
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from collections import defaultdict

from transformers import AutoTokenizer

sys.path.append('../..')
sys.path


GREATEST_INTEGER = sys.maxsize

def retorna_num_tokens(parm_texto:str, parm_tokenizador:AutoTokenizer):
    return len(parm_tokenizador.tokenize(parm_texto))

# Function to invert keys and values keeping a list of keys for each value
def invert_dict_with_lists(d):
    inverted_dict = defaultdict(list)
    for k, v in d.items():
        inverted_dict[v].append(k)
    return dict(inverted_dict)


def return_consolidate_result(parm_dataset):
    path_search_result_consolidated = f'../../data/search/{parm_dataset}/search_result_consolidated_{parm_dataset}.csv'
    df_result = pd.read_csv(path_search_result_consolidated)
    df_result['LIST_DOCTO_FOUND'] = df_result['LIST_DOCTO_FOUND'].apply(ast.literal_eval)
    df_result['LIST_RANK'] = df_result['LIST_RANK'].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else [])
    df_result['RELEVANCE_DICT_ID_DOCTO'] = df_result['RELEVANCE_DICT_ID_DOCTO'].apply(ast.literal_eval)
    df_result['RELEVANCE_DICT_TYPE'] = df_result['RELEVANCE_DICT_TYPE'].apply(ast.literal_eval)


    # calculate redundant fields
    df_result['COUNT_DOCTO_RELEVANT_FOUND'] = df_result['LIST_RANK'].apply(len)
    df_result['PERCENT_DOCTO_RELEVANT_FOUND'] = round(100 * df_result['COUNT_DOCTO_RELEVANT_FOUND']  / df_result['COUNT_DOCTO_RELEVANT'], 2)
    df_result['RANKER_TYPE'] = df_result['RANKER_MODEL_NAME'].apply(lambda x: 'monott5' if 'mt5' in x else 'minilm')

    return df_result

def consolidate_result(parm_dataset):

    path_search_experiment =  f'../../data/search/{parm_dataset}/search_experiment_{parm_dataset}.csv'
    path_search_result =  f'../../data/search/{parm_dataset}/search_experiment_result_{parm_dataset}.csv'
    path_query = f'../../data/{parm_dataset}/query.csv'
    path_qrel =  f'../../data/{parm_dataset}/qrel.csv'
    path_search_result_consolidated = f'../../data/search/{parm_dataset}/search_result_consolidated_{parm_dataset}.csv'


    # read experiment data
    df_experiment = pd.read_csv(path_search_experiment)
    df_experiment_result = pd.read_csv(path_search_result)
    print(f'df_experiment.shape[0] {df_experiment.shape[0]}')
    print(f'df_experiment_result.shape[0] {df_experiment_result.shape[0]}')

    # merge experiment data
    df_experiment_result = df_experiment_result.merge(df_experiment, how='inner', on='TIME')
    print(f'df_experiment_result.shape {df_experiment_result.shape}')
    df_experiment_result.drop(['RANK1_MEAN', 'NDCG_MEAN','TIME_SPENT_MEAN'], axis=1, inplace=True)

    # read query/qrel data
    df_query = pd.read_csv(path_query)
    print(f'df_query.shape[0] {df_query.shape[0]}')

    df_qrel = pd.read_csv(path_qrel)
    print(f'df_qrel.shape[0] {df_qrel.shape[0]}')

    df_search_data = df_query.merge(df_qrel, how='left', left_on='ID', right_on='ID_QUERY').drop('ID_QUERY', axis=1)
    print(f'df_search_data.shape[0] {df_search_data.shape[0]}')


    # consolidate data of queries
    df_new = df_search_data.groupby('ID').apply(lambda x: dict(zip(x['ID_DOCTO'], x['TYPE']))).reset_index(name='RELEVANCE_DICT_ID_DOCTO')
    df_new['RELEVANCE_DICT_TYPE'] = df_new['RELEVANCE_DICT_ID_DOCTO'].apply(invert_dict_with_lists)
    df_new = pd.merge(df_new, df_search_data.drop_duplicates('ID'), on='ID', how='left')

    # select desired columns
    # df_search_data = df_new[['ID', 'TEXT', 'REFERENCE_LIST', 'PARADIGMATIC', 'AREA_NAME', 'AREA_ID_DESCRIPTOR', 'NORMATIVE_PROCESS_TYPE', 'NORMATIVE_IDENTIFICATION', 'NORMATIVE_DATE', 'NORMATIVE_AUTHOR_TYPE', 'NORMATIVE_AUTHOR_NAME', 'RELEVANCE_DICT_ID_DOCTO','RELEVANCE_DICT_TYPE']]


    # merge experiment data with queries searched
    df_experiment_result = df_experiment_result.merge(df_new, how='inner', left_on='ID_QUERY', right_on='ID')


    # adding text lenght info
    nome_modelo_ranking_pt = 'unicamp-dl/mMiniLM-L6-v2-pt-v2'
    nome_caminho_modelo_pt = "/home/borela/fontes/relevar-busca/modelo/" + nome_modelo_ranking_pt
    assert os.path.exists(nome_caminho_modelo_pt), f"Path para {nome_caminho_modelo_pt} não existe!"
    nome_modelo_monot5_3b = 'unicamp-dl/mt5-3B-mmarco-en-pt'
    # "A mono-ptT5 reranker model (850 mb) pretrained in the BrWac corpus, finetuned for 100k steps on Portuguese translated version of MS MARCO passage dataset. The portuguese dataset was translated using Google Translate.")
    nome_caminho_modelo_3b = "/home/borela/fontes/relevar-busca/modelo/" + nome_modelo_monot5_3b
    assert os.path.exists(nome_caminho_modelo_3b), f"Path para {nome_caminho_modelo_3b} não existe!"
    tokenizador_pt_monot5_3b = AutoTokenizer.from_pretrained(nome_caminho_modelo_3b)
    tokenizador_pt_minilm = AutoTokenizer.from_pretrained(nome_caminho_modelo_pt)

    df_experiment_result['LEN_TEXT_CHAR'] = df_experiment_result['TEXT'].apply(len)
    df_experiment_result['LEN_TEXT_CHAR_LOG'] = round(np.log(df_experiment_result['TEXT'].apply(len))).astype(int)
    df_experiment_result['NUM_WORD'] = df_experiment_result['TEXT'].apply(lambda x: len(x.split()))
    df_experiment_result['NUM_TOKENS_MONOT5_3B'] = df_experiment_result['TEXT'].apply(retorna_num_tokens, parm_tokenizador=tokenizador_pt_monot5_3b)
    df_experiment_result['NUM_TOKENS_MINILM'] = df_experiment_result['TEXT'].apply(retorna_num_tokens, parm_tokenizador=tokenizador_pt_minilm)
    df_experiment_result['NUM_TOKENS_MONOT5_3B_LOG'] = round(np.log(df_experiment_result['NUM_TOKENS_MONOT5_3B'])).astype(int)
    df_experiment_result['NUM_TOKENS_MINILM_LOG'] = round(np.log(df_experiment_result['NUM_TOKENS_MINILM'])).astype(int)


    # saving data
    columns_to_remove = [# about experiment data
                        'TIME', 'COUNT_QUERY_RUN',
                         # about query/qrel data
                         'TEXT', 'ID', 'TYPE', 'REFERENCE_LIST', 'AREA_ID_DESCRIPTOR', 'NORMATIVE_PROCESS_TYPE', 'NORMATIVE_IDENTIFICATION', 'NORMATIVE_DATE', 'NORMATIVE_AUTHOR_TYPE', 'NORMATIVE_AUTHOR_NAME']
    #print([x for x in columns_to_remove if x not in df_experiment_result.columns])
    df_experiment_result.drop(columns_to_remove, axis=1, inplace=True)
    df_experiment_result.to_csv(path_search_result_consolidated, index=False)
    print(f"Generated file with {df_experiment_result.shape[0]} records")


def generate_dict_idcg(count_qrel_max: int = 15, val_relevance: int = 1):
    """
    Generate a dictionary of IDCG (Ideal Discounted Cumulative Gain) values.

    Args:
        count_qrel_max (int): Maximum value for the IDCG calculation (exclusive).
        val_relevance (int): Relevance value used in the IDCG calculation.

    Returns:
        dict: A dictionary where keys represent the position (ranging from 1 to count_qrel_max - 1)
              and values represent the corresponding IDCG value.

    """
    dict_idcg_relevance_fixed = {}

    # Iterate from 1 to count_qrel_max - 1
    for i in range(1, count_qrel_max):

        idcg = 0

        # Iterate from 0 to i - 1
        for j in range(i):

            # Calculate the IDCG value based on the provided relevance value
            idcg += (2 ** val_relevance - 1) / math.log2(j + 2)

        # Store the calculated IDCG value in the dictionary
        dict_idcg_relevance_fixed[i] = idcg

    return dict_idcg_relevance_fixed


def calculate_ndcg_query_result (list_id_doc_returned, dict_doc_relevant:dict, position:int)->list:
    """
    list_id_doc_returned: list of id of documents returned in search
    dict_doc_relevant: dict of relevance where id_doc is the key and type relevance (not used) value
    return ndcg at position
    """
    if len(list_id_doc_returned) > 0:

        # calculating dcg for the query (Discounted Cumulative Gain)
        dcg = 0
        for i, docid in enumerate(list_id_doc_returned[:position]):
            if i >= position:
                raise ValueError('Logic error: more than position????')
            relevance = 1 if docid in dict_doc_relevant else 0
            dcg += (2 ** relevance - 1) / math.log2(i + 2)

        # calculate ndcg for the query (Normalized Discounted Cumulative Gain)
        ndcg = dcg / dict_idcg_relevance_fixed[len(dict_doc_relevant)]

        return ndcg
    else:
        return 0

def calculate_list_rank_query_result (list_id_doc_returned, dict_doc_relevant:dict)->list:
    """
    list_id_doc_returned: list of id of documents returned in search
    dict_doc_relevant: dict of relevance where id_doc is the key and type relevance (not used) value
    return min (and mean) rank  of relevant docto
    """


    if len(list_id_doc_returned) > 0:
        list_rank = []
        for id_docto in dict_doc_relevant:
            if id_docto in list_id_doc_returned:
                list_rank.append(1 + list_id_doc_returned.index(id_docto))  # 1st position of id_docto in the list
        return list_rank
    else:
        return None


dict_idcg_relevance_fixed = generate_dict_idcg(15, 1)

# print('dict_idcg_relevance_fixed', dict_idcg_relevance_fixed)
