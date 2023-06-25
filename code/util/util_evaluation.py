
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

from util.util_pipeline_v2 import dict_ranker

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



def find_ranker_type(row):
    ranker_model_name = row['RANKER_MODEL_NAME']
    if pd.isnull(ranker_model_name) or ranker_model_name == "":
        return 'none'
    for key, value in dict_ranker.items():
        if isinstance(ranker_model_name, str) and ranker_model_name.lower() in value['model_name'].lower():
            return key
    return 'unknown'


def return_consolidate_result(parm_dataset):
    path_search_result_consolidated = f'../data/search/{parm_dataset}/search_result_consolidated_{parm_dataset}.csv'
    df_result = pd.read_csv(path_search_result_consolidated)
    df_result['LIST_DOCTO_RETURNED'] = df_result['LIST_DOCTO_RETURNED'].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else [])
    df_result['LIST_RANK'] = df_result['LIST_RANK'].apply(lambda x: ast.literal_eval(x) if not pd.isna(x) else [])
    df_result['QUERY_RELEVANCE_DICT_ID_DOC'] = df_result['QUERY_RELEVANCE_DICT_ID_DOC'].apply(ast.literal_eval)
    df_result['QUERY_RELEVANCE_DICT_TYPE'] = df_result['QUERY_RELEVANCE_DICT_TYPE'].apply(ast.literal_eval)


    # calculate redundant fields
    df_result['COUNT_DOCTO_RELEVANT_FOUND'] = df_result['LIST_RANK'].apply(len)
    df_result['PERCENT_DOCTO_RELEVANT_FOUND'] = round(100 * df_result['COUNT_DOCTO_RELEVANT_FOUND']  / df_result['COUNT_DOCTO_RELEVANT'], 2)
    # df_result['RANKER_TYPE'] = df_result['RANKER_MODEL_NAME'].apply(lambda x: 'none' if pd.isnull(x) else 'none' if x=="" else 'monot5' if isinstance(x, str) and 'mt5' in x.lower() else 'minilm' if isinstance(x, str) and 'minilm' in x.lower() else 'unknown')
    df_result['RANKER_TYPE'] = df_result.apply(find_ranker_type, axis=1)

    return df_result

def consolidate_result(parm_dataset):

    path_search_experiment =  f'../data/search/{parm_dataset}/search_experiment_{parm_dataset}.csv'
    path_search_result =  f'../data/search/{parm_dataset}/search_experiment_result_{parm_dataset}.csv'
    path_query = f'../data/{parm_dataset}/query.csv'
    path_qrel =  f'../data/{parm_dataset}/qrel.csv'
    path_search_result_consolidated = f'../data/search/{parm_dataset}/search_result_consolidated_{parm_dataset}.csv'


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
    #df_search_data.rename(columns={'TEXT': 'QUERY_TEXT', 'ID':'ID_QUERY'},inplace=True)

    # consolidate data of queries
    df_new = df_search_data.groupby('ID').apply(lambda x: dict(zip(x['ID_DOCTO'], x['TYPE']))).reset_index(name='RELEVANCE_DICT_ID_DOC')
    df_new['RELEVANCE_DICT_TYPE'] = df_new['RELEVANCE_DICT_ID_DOC'].apply(invert_dict_with_lists)
    df_new = pd.merge(df_new, df_search_data.drop_duplicates('ID'), on='ID', how='left')
    df_new = df_new.add_prefix('QUERY_')
    # print('após add prefix', df_new.columns)


    # merge experiment data with queries searched
    df_experiment_result = df_experiment_result.merge(df_new, how='inner', left_on='ID_QUERY', right_on='QUERY_ID').drop('ID_QUERY', axis=1)


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

    df_experiment_result['QUERY_LEN_TEXT_CHAR'] = df_experiment_result['QUERY_TEXT'].apply(len)
    df_experiment_result['QUERY_LEN_TEXT_CHAR_LOG'] = round(np.log(df_experiment_result['QUERY_TEXT'].apply(len))).astype(int)
    df_experiment_result['QUERY_NUM_WORD'] = df_experiment_result['QUERY_TEXT'].apply(lambda x: len(x.split()))
    df_experiment_result['QUERY_NUM_TOKENS_MONOT5_3B'] = df_experiment_result['QUERY_TEXT'].apply(retorna_num_tokens, parm_tokenizador=tokenizador_pt_monot5_3b)
    df_experiment_result['QUERY_NUM_TOKENS_MINILM'] = df_experiment_result['QUERY_TEXT'].apply(retorna_num_tokens, parm_tokenizador=tokenizador_pt_minilm)
    df_experiment_result['QUERY_NUM_TOKENS_MONOT5_3B_LOG'] = round(np.log(df_experiment_result['QUERY_NUM_TOKENS_MONOT5_3B'])).astype(int)
    df_experiment_result['QUERY_NUM_TOKENS_MINILM_LOG'] = round(np.log(df_experiment_result['QUERY_NUM_TOKENS_MINILM'])).astype(int)

    # saving data
    columns_to_remove = [# about experiment data
                        # 'TIME', # 'COUNT_QUERY_RUN',
                         # about query/qrel data
                         'QUERY_TEXT', 'QUERY_TYPE', 'QUERY_REFERENCE_LIST', 'QUERY_AREA_ID_DESCRIPTOR', 'QUERY_NORMATIVE_PROCESS_TYPE', 'QUERY_NORMATIVE_IDENTIFICATION',
                         'QUERY_NORMATIVE_DATE', 'QUERY_NORMATIVE_AUTHOR_TYPE', 'QUERY_NORMATIVE_AUTHOR_NAME',
                         ]
    #print([x for x in columns_to_remove if x not in df_experiment_result.columns])
    df_experiment_result.drop(columns_to_remove, axis=1, inplace=True)
    df_experiment_result.to_csv(path_search_result_consolidated, index=False)
    print(f"Generated file with {df_experiment_result.shape[0]} records")


def return_result_per_doc(parm_dataset):
    path_doc = f'../data/{parm_dataset}/doc.csv'
    df_result = return_consolidate_result(parm_dataset)

    # Criar uma lista para armazenar os registros do novo DataFrame
    result_doc_records = []
    for ndx, item in df_result.iterrows():
        # Iterar sobre cada par chave-valor no item
        for k, v in item['QUERY_RELEVANCE_DICT_ID_DOC'].items():
            # Criar um novo registro com todas as colunas de df_result e as colunas adicionais doc_id e type_qrel
            new_record = item.copy()
            new_record['DOC_ID'] = k  # Adicionar o valor k em doc_id
            new_record['TYPE_QREL'] = v  # Adicionar o valor v em type_qrel
            result_doc_records.append(new_record)

    # Criar o novo DataFrame df_result_doc
    df_result_doc = pd.DataFrame(result_doc_records)


    # Excluir a coluna QUERY_RELEVANCE_DICT_ID_DOC de df_result_doc
    df_result_doc = df_result_doc.drop(['QUERY_RELEVANCE_DICT_ID_DOC', 'QUERY_RELEVANCE_DICT_TYPE'], axis=1)

    df_result_doc['DOC_ID_FOUND'] = df_result_doc.apply(lambda row: row['DOC_ID'] in row['LIST_DOCTO_FOUND'], axis=1)
    df_doc = pd.read_csv(path_doc).drop(['TEXT_DEFINITION',
        'TEXT_SYNONYM', 'TEXT_RELATED_TERM', 'TEXT_SCOPE_NOTE', 'TEXT_EXAMPLE',
        'TEXT_ENGLISH_TRANSLATION', 'TEXT_SPANISH_TRANSLATION',
        'TEXT_SPECIALIZATION', 'TEXT_GENERALIZATION', 'DATE_REFERENCE'], axis=1)
    # df_doc.rename(columns={'TEXT': 'DOC_TEXT', 'ID':'ID_DOC'},inplace=True)
    df_doc = df_doc.add_prefix('DOC_')
    df_result_doc = df_result_doc.merge(df_doc, how='inner', on='DOC_ID')



    df_result_doc['DOC_LEN_TEXT_CHAR'] = df_result_doc['DOC_TEXT'].apply(len)
    df_result_doc['DOC_LEN_TEXT_CHAR_LOG'] = round(np.log(df_result_doc['DOC_TEXT'].apply(len))).astype(int)
    df_result_doc['DOC_NUM_WORD'] = df_result_doc['DOC_TEXT'].apply(lambda x: len(x.split()))

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



    df_result_doc['DOC_NUM_TOKENS_MONOT5_3B'] = df_result_doc['DOC_TEXT'].apply(retorna_num_tokens, parm_tokenizador=tokenizador_pt_monot5_3b)
    df_result_doc['DOC_NUM_TOKENS_MINILM'] = df_result_doc['DOC_TEXT'].apply(retorna_num_tokens, parm_tokenizador=tokenizador_pt_minilm)
    df_result_doc['DOC_NUM_TOKENS_MONOT5_3B_LOG'] = round(np.log(df_result_doc['DOC_NUM_TOKENS_MONOT5_3B'])).astype(int)
    df_result_doc['DOC_NUM_TOKENS_MINILM_LOG'] = round(np.log(df_result_doc['DOC_NUM_TOKENS_MINILM'])).astype(int)


    # saving data
    columns_to_remove = [
                         'DOC_TEXT'
                         ]
    #print([x for x in columns_to_remove if x not in df_experiment_result.columns])
    df_result_doc.drop(columns_to_remove, axis=1, inplace=True)


    return df_result_doc

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
