
"""
rotinas de cálculo de métrica
"""
import sys
import os
import time
import pandas as pd
from tqdm import tqdm
import math
import numpy as np


import logging
logging.getLogger("haystack").setLevel(logging.WARNING) #WARNING, INFO


# interesting links
#     https://docs.haystack.deepset.ai/docs/metadata-filtering#filtering-logic

GREATEST_INTEGER = sys.maxsize

def generate_dict_idcg(count_qrel_max: int = 15):
    """
    Generate a dictionary of IDCG (Ideal Discounted Cumulative Gain) values.

    Args:
        count_qrel_max (int): Maximum value for the IDCG calculation (exclusive).

    Returns:
        dict: A dictionary where keys represent the position (ranging from 1 to count_qrel_max - 1)
              and values represent the corresponding IDCG value.

    Obs.: Assumido val_relevance (int) == 1 Relevance value used in the IDCG calculation.

    """
    dict_idcg_relevance_fixed = {}

    # Iterate from 1 to count_qrel_max - 1
    for i in range(1, count_qrel_max):

        idcg = 0

        # Iterate from 0 to i - 1
        for j in range(i):

            # Calculate the IDCG value based on the provided relevance value
            # idcg += (2 ** val_relevance - 1) / math.log2(j + 2)
            idcg +=  1 / math.log2(j + 2)

        # Store the calculated IDCG value in the dictionary
        dict_idcg_relevance_fixed[i] = idcg

    return dict_idcg_relevance_fixed

def calculate_ndcg_query_result (list_id_doc_returned, dict_doc_relevant:dict, position:int)->list:
    """
    list_id_doc_returned: list of id of documents returned in search
    dict_doc_relevant: dict of relevance where id_doc is the key and type relevance (not used) value
    return ndcg at position
    """
    global dict_idcg_relevance_fixed
    if list_id_doc_returned is not None and len(list_id_doc_returned) > 0:

        # calculating dcg for the query (Discounted Cumulative Gain)
        dcg = 0
        for i, docid in enumerate(list_id_doc_returned[:position]):
            if i >= position:
                raise ValueError('Logic error: more than position????')
            relevance = 1 if docid in dict_doc_relevant else 0
            dcg += (2 ** relevance - 1) / math.log2(i + 2)

        # if len(dict_doc_relevant) not in dict_idcg_relevance_fixed:
        #     raise Exception(f"len(dict_doc_relevant) {len(dict_doc_relevant)} not in dict_idcg_relevance_fixed {dict_idcg_relevance_fixed}")
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


    if list_id_doc_returned is not None and len(list_id_doc_returned) > 0:
        list_rank = []
        for id_docto in dict_doc_relevant:
            if id_docto in list_id_doc_returned:
                list_rank.append(1 + list_id_doc_returned.index(id_docto))  # 1st position of id_docto in the list
        return list_rank
    else:
        return None

def build_search_parameter(parm_experiment, query_data):
    dict_param_busca = {}
    criteria_type = dict_criterio[parm_experiment['CRITERIA']]['type']
    criteria_value = dict_criterio[parm_experiment['CRITERIA']]['value']
    if  'join' in parm_experiment['PIPE']['RETRIEVER_TYPE']:
        dict_param_busca = {"Bm25Retriever":{"top_k": np.int(parm_experiment['TOPK_RETRIEVER']/2), "filters": criteria_value},
                            "StsRetriever": {"top_k": np.int(parm_experiment['TOPK_RETRIEVER']/2), "filters": criteria_value}}
    elif parm_experiment['PIPE']['RETRIEVER_TYPE'] in ['bm25','sts','sts_multihop']:
        if criteria_type == "static":
            dict_param_busca = {"Retriever":{"top_k": parm_experiment['TOPK_RETRIEVER'], "filters": criteria_value}}
        else:
            raise Exception (f"Invalid criteria_type  {criteria_type} in build_search_parameter")
    else:
        raise Exception (f"Invalid parm_experiment['PIPE']['RETRIEVER_TYPE']  {parm_experiment['PIPE']['RETRIEVER_TYPE'] }")
    if 'TOPK_RANKER' in parm_experiment and parm_experiment['TOPK_RANKER'] > 0:
        dict_param_busca.update({"Ranker":{"top_k": parm_experiment['TOPK_RANKER']}})
    return dict_param_busca

def search_docto_for_experiment(parm_experiment, query_data):
    """
    Busca documentos conforme parâmetros
    """
    limit_size_text_first_stage = 1024
    if parm_experiment['CRITERIA'] is not None and parm_experiment['PIPE']['RETRIEVER_TYPE'] == 'TFIDF':
        raise Exception (f'Retriever tfidf do not combine with filters')
    search_parameter = build_search_parameter(parm_experiment, query_data)
    # print(f"search_parameter: {search_parameter} ")
    search_text =  query_data['TEXT'][:limit_size_text_first_stage]
    docto_found = parm_experiment['PIPE']['PIPE_OBJECT'].run(query=search_text, params=search_parameter)
    if 'documents' in docto_found: # search with pipe
        if len(docto_found['documents']) > 0:
            #if isinstance(docto_found['documents'][0], dict):
            #    list_id_doc_returned = [int(docto['id']) for docto in docto_found['documents']]
            #else:
            list_id_doc_returned = [docto.meta['id'] for docto in docto_found['documents']]
            return list_id_doc_returned #, search_parameter
    elif len(docto_found) > 0: # search with retriever
        #if isinstance(docto_found[0], dict):
        #    list_id_doc_returned = [int(docto['id']) for docto in docto_found]
        #else:
        list_id_doc_returned = [docto.meta['id'] for docto in docto_found]
        return list_id_doc_returned # , search_parameter
    else:
        return None

def calculate_ground_truth(parm_criteria:str, parm_dict:dict)->dict:
    if "total" in parm_criteria or "class_termo" in parm_criteria:
        return parm_dict
    # Considera os valores do tipo de indexação armazenados
    elif parm_criteria == "area":
        return {key: value for key, value in parm_dict.items() if value == 'AREA'}
    elif parm_criteria == "theme":
        return {key: value for key, value in parm_dict.items() if value == 'TEMA'}
    elif parm_criteria == "subtheme":
        return {key: value for key, value in parm_dict.items() if value == 'SUBTEMA'}
    elif parm_criteria == "extra":
        return {key: value for key, value in parm_dict.items() if value == 'INDEXACAO_EXTRA'}
    else:
        raise Exception(f"Invalid parm_criteria {parm_criteria} in calculate_ground_truth")

def experiment_run(parm_df,  parm_experiment,
                    parm_limit_query:int=GREATEST_INTEGER,
                    parm_ndcg_position:int=12,
                    parm_print:bool=False):
    """
    Consider run as search for all queries
    """

    # param validation
    list_experiment_keys_expected = ['CRITERIA', 'TOPK_RETRIEVER', 'TOPK_RANKER', 'PIPE', 'DONE']
    list_pipe_keys_expected = ['PIPE_NAME', 'PIPE_OBJECT', 'RETRIEVER_TYPE', 'RETRIEVER_MODEL_NAME', 'RANKER_MODEL_NAME']

    # Verificar se todas as chaves de list_keys estão em parm_experiment
    if set(list_experiment_keys_expected) != set(parm_experiment.keys()):
        raise Exception(f"Invalid keys in parm_experiment {parm_experiment.keys()}. Expected: {list_experiment_keys_expected}")

    # Verificar se todas as chaves de list_keys estão em parm_experiment
    if set(list_pipe_keys_expected) != set(parm_experiment['PIPE'].keys()):
        raise Exception(f"Invalid keys in parm_experiment['PIPE'] {parm_experiment['PIPE'].keys()}. Expected: {list_pipe_keys_expected}")

    if parm_limit_query==100:
        PATH_VALID_DATA = f"../data/train_juris_tcu_index/juris_tcu_index_validation_query_id.csv"
        df_validation = pd.read_csv(PATH_VALID_DATA)
        list_query_id_valid = df_validation['QUERY_ID'].to_list()
        df = parm_df[parm_df['ID'].isin(list_query_id_valid)]
        print(f"Experimento envolverá {df.shape[0]} registros, queries selecionadas para validação")
    elif parm_limit_query<parm_df.shape[0]:
        randomness_fraction = count_query_run/len(parm_df)
        print(f"Experimento envolverá {df.shape[0]} registros com randomness_fraction: {randomness_fraction}")
        df = parm_df.sample(frac=randomness_fraction, random_state=123).reset_index(drop=True)
    else:
        print(f"Experimento envolverá {df.shape[0]} registros, toda a base de dados")
        df = parm_df
    count_query_run = min(df.shape[0],parm_limit_query)
    result_search_all_query = []
    total_rank1 = 0 #
    total_ndcg = 0
    total_without_result = 0
    total_found = 0 # total de buscas em que se encantrou algum documento relevante na lista retornada
    total_not_found = 0
    time_start_search_run = time.time()
    count = 0
    for ndx, row_query in tqdm(df.iterrows(), mininterval=10, total=count_query_run):
        count += 1
        if count > count_query_run:
            raise Exception('Stoped before data end?')
        result_search_one_query = {}
        result_search_one_query['ID_QUERY'] = row_query['ID']
        time_start_search_query = time.time()
        list_id_doc_returned = search_docto_for_experiment(parm_experiment=parm_experiment,  query_data=row_query)
        result_search_one_query['TIME_SPENT'] = round((time.time() - time_start_search_query),4)
        if list_id_doc_returned is None or len(list_id_doc_returned) == 0:
            print(f"\nDocuments not found in experiment {parm_experiment} With query: {row_query['ID']}")
            result_search_one_query['RANK1'] = 0
            result_search_one_query['NDCG'] = 0
            result_search_one_query['COUNT_DOCTO_FOUND'] = 0
            result_search_one_query['COUNT_DOCTO_RELEVANT'] = 0
            result_search_one_query['LIST_RANK'] = ""
            result_search_one_query['LIST_DOCTO_RETURNED'] = ""
            total_without_result += 1
        else:
            result_search_one_query['LIST_DOCTO_RETURNED'] = str(list_id_doc_returned)
            dict_ground_truth = calculate_ground_truth(parm_criteria=parm_experiment['CRITERIA'], parm_dict=row_query['RELEVANCE_DICT'])
            if len(dict_ground_truth) == 0:
                if parm_experiment['CRITERIA'] != 'extra':
                    raise Exception (f"no value for calculate_ground_truth, but parm_experiment['CRITERIA'] {parm_experiment['CRITERIA']} != 'extra'")
                # pode não haver ground truth dependendo do critério, por exemplo, não haver indexação extra
                result_search_one_query['RANK1'] = 0
                result_search_one_query['NDCG'] = 0
                result_search_one_query['COUNT_DOCTO_FOUND'] = 0
                result_search_one_query['COUNT_DOCTO_RELEVANT'] = 0
                result_search_one_query['LIST_RANK'] = ""
                total_not_found += 1
            else:
                list_rank = calculate_list_rank_query_result(list_id_doc_returned, dict_ground_truth)
                if list_rank is None or len(list_rank) == 0:
                    result_search_one_query['RANK1'] = 0
                    result_search_one_query['NDCG'] = 0
                    result_search_one_query['COUNT_DOCTO_FOUND'] = 0
                    result_search_one_query['COUNT_DOCTO_RELEVANT'] = 0
                    result_search_one_query['LIST_RANK'] = ""
                    total_not_found += 1
                else:
                    result_search_one_query['NDCG'] = round(100*calculate_ndcg_query_result(list_id_doc_returned, dict_ground_truth, parm_ndcg_position),2)
                    result_search_one_query['COUNT_DOCTO_FOUND'] = len(list_id_doc_returned) if list_id_doc_returned is not None else 0
                    result_search_one_query['COUNT_DOCTO_RELEVANT'] = len(dict_ground_truth)
                    result_search_one_query['GROUND_TRUTH'] = str(dict_ground_truth)
                    result_search_one_query['RANK1'] = min(list_rank)
                    result_search_one_query['LIST_RANK'] = str(list_rank)
                    total_found += 1
        # convertendo campos do tipo lista em string

        total_rank1 += result_search_one_query['RANK1']
        total_ndcg += result_search_one_query['NDCG']
        result_search_all_query.append(result_search_one_query)


    total_time = time.time() - time_start_search_run
    result_search_run = {}
    result_search_run['TIME'] = time.strftime('%Y-%b-%d %H:%M:%S')
    result_search_run['RETRIEVER_TYPE'] = parm_experiment['PIPE']['RETRIEVER_TYPE']
    result_search_run['COUNT_QUERY_RUN'] = count_query_run
    result_search_run['TOPK_RETRIEVER'] = parm_experiment['TOPK_RETRIEVER']
    result_search_run['TOPK_RANKER'] = parm_experiment['TOPK_RANKER']
    result_search_run['COUNT_QUERY_RUN'] = count_query_run
    result_search_run['COUNT_QUERY_WITHOUT_RESULT'] = total_without_result
    result_search_run['COUNT_QUERY_NOT_FOUND'] = total_not_found
    if total_found > 0:
        result_search_run['RANK1_MEAN'] = round(total_rank1/total_found,3)
    else:
        result_search_run['RANK1_MEAN'] = 0
    result_search_run['NDCG_MEAN'] = round(total_ndcg/count_query_run,3)
    result_search_run['NDCG_LIMIT'] = parm_ndcg_position
    result_search_run['TIME_SPENT_MEAN'] = round(total_time/count_query_run,3)
    result_search_run['CRITERIA'] = parm_experiment['CRITERIA']
    result_search_run['RETRIEVER_MODEL_NAME'] = parm_experiment['PIPE']['RETRIEVER_MODEL_NAME']
    result_search_run['RANKER_MODEL_NAME'] = parm_experiment['PIPE']['RANKER_MODEL_NAME']
    result_search_run['RESULT_QUERY'] = result_search_all_query

    if parm_print: # print results
        for key in ['RANK1_MEAN','NDCG_MEAN','TIME_SPENT_MEAN','COUNT_QUERY_RUN', 'COUNT_QUERY_WITHOUT_RESULT','COUNT_QUERY_NOT_FOUND']:
            print(f"{key:>8}: {result_search_run[key]}")

    return result_search_run

def del_experiment_value_column(column_name, column_value, parm_dataset, parm_confirm=True):
    # Load the dataframes

    print(f"Excluindo {column_name} = {column_value}")
    path_search_experiment =  f'../data/search/{parm_dataset}/search_experiment_{parm_dataset}.csv'
    path_search_result =  f'../data/search/{parm_dataset}/search_experiment_result_{parm_dataset}.csv'


    df_experiment = pd.read_csv(path_search_experiment)
    df_experiment_result = pd.read_csv(path_search_result)

    # Count the records with a value equal to column_value in the column_name
    count_filtered_experiment = df_experiment[df_experiment[column_name] == column_value].shape[0]


    # Count the records with a value equal to column_value in the column_name and also in the TIME column
    # Get the values of the TIME column to be deleted from df_experiment
    time_values_to_delete = df_experiment[df_experiment[column_name] == column_value]['TIME'].tolist()
    count_filtered_experiment_result = df_experiment_result[df_experiment_result['TIME'].isin(time_values_to_delete)].shape[0]

    # Check if there are records to be deleted
    if count_filtered_experiment == 0 and count_filtered_experiment_result == 0:
        print("There are no records to be deleted.")
        return

    # Display the records to be deleted
    print(f"Records to be deleted in df_experiment: {count_filtered_experiment}")
    print(f"Records to be deleted in df_experiment_result: {count_filtered_experiment_result}")

    if parm_confirm:
        # Ask for user confirmation
        confirmation = input("Do you really want to delete the records? (y/n): ")
        if confirmation.lower() != 'y':
            print("Operation canceled by the user.")
            return

    # Delete the records
    df_experiment = df_experiment[df_experiment[column_name] != column_value]
    df_experiment_result = df_experiment_result[~df_experiment_result['TIME'].isin(time_values_to_delete)]

    df_experiment.to_csv(path_search_experiment, sep=',', index=False)
    df_experiment_result.to_csv(path_search_result, sep=',', index=False)

    print("Records successfully deleted.")

def del_experiment_result(time_key, parm_dataset):
    # Load the dataframes

    path_search_experiment =  f'../data/search/{parm_dataset}/search_experiment_{parm_dataset}.csv'
    path_search_result =  f'../data/search/{parm_dataset}/search_experiment_result_{parm_dataset}.csv'


    df_experiment = pd.read_csv(path_search_experiment)
    df_experiment_result = pd.read_csv(path_search_result)

    # Filter the records with a value equal to time_key in the TIME column
    count_filtered_experiment = df_experiment[df_experiment['TIME'] == time_key].shape[0]
    count_filtered_experiment_result = df_experiment_result[df_experiment_result['TIME'] == time_key].shape[0]

    # Check if there are records to be deleted
    if count_filtered_experiment == 0 and count_filtered_experiment_result == 0:
        print("There are no records to be deleted.")
        return

    # Display the records to be deleted
    print(f"Records to be deleted in df_experiment: {count_filtered_experiment}")
    print(f"Records to be deleted in df_experiment_result: {count_filtered_experiment_result}")

    # Ask for user confirmation
    confirmation = input("Do you really want to delete the records? (y/n): ")
    if confirmation.lower() != 'y':
        print("Operation canceled by the user.")
        return

    # Delete the records
    df_experiment = df_experiment[df_experiment['TIME'] != time_key]
    df_experiment_result = df_experiment_result[df_experiment_result['TIME'] != time_key]

    df_experiment.to_csv(path_search_experiment, sep=',', index=False)
    df_experiment_result.to_csv(path_search_result, sep=',', index=False)

    print("Records successfully deleted.")


def add_experiment_result(parm_list_result, parm_dataset):

    path_search_experiment =  f'../data/search/{parm_dataset}/search_experiment_{parm_dataset}.csv'
    path_search_result =  f'../data/search/{parm_dataset}/search_experiment_result_{parm_dataset}.csv'

    # criar um dataframe pandas a partir do dicionário
    df_experiment = pd.DataFrame(parm_list_result)

    # Inicialize o dataframe vazio para a saída
    df_experiment_result = pd.DataFrame()

    # Itere sobre cada linha da coluna METRICA_TERMOS e adicione ao dataframe de saída
    for _, row_query in df_experiment.iterrows():
        momento = row_query['TIME']
        for query_search_result in row_query['RESULT_QUERY']:
            # print(f'momento {momento}')
            # print(f'query_search_result {query_search_result}')
            query_search_result['TIME'] = momento
            df_experiment_result = df_experiment_result.append(query_search_result, ignore_index=True)
    del df_experiment['RESULT_QUERY']
    df_experiment_result['ID_QUERY'] = df_experiment_result['ID_QUERY'].astype(int)
    df_experiment_result['RANK1'] = df_experiment_result['RANK1'].astype(int)
    df_experiment_result['COUNT_DOCTO_FOUND'] = df_experiment_result['COUNT_DOCTO_FOUND'].astype(int)
    df_experiment_result['COUNT_DOCTO_RELEVANT'] = df_experiment_result['COUNT_DOCTO_RELEVANT'].astype(int)


    if not os.path.exists(path_search_experiment):
        df_experiment.to_csv(path_search_experiment, sep = ',', index=False)


        # Salvando coluna TIME no início
        # Lista de colunas existentes no DataFrame
        colunas = df_experiment_result.columns.tolist()
        # Move a coluna "TIME" para o início
        colunas.insert(0, colunas.pop(colunas.index('TIME')))
        # Reordena as colunas do DataFrame
        df_experiment_result = df_experiment_result[colunas]

        df_experiment_result.to_csv(path_search_result, sep = ',', index=False)
    else: # concatenar a arquivo existente
        # Leitura do arquivo CSV existente
        df_experiment_save = pd.read_csv(path_search_experiment)

        # Concatenação dos dataframes
        df_experiment = pd.concat([df_experiment_save, df_experiment], ignore_index=True)

        # Salvando o dataframe concatenado no arquivo CSV
        df_experiment.to_csv(path_search_experiment, sep = ',', index=False)

        # Leitura do arquivo CSV existente
        df_experiment_result_save = pd.read_csv(path_search_result)

        # Concatenação dos dataframes
        df_experiment_result = pd.concat([df_experiment_result_save, df_experiment_result], ignore_index=True)

        df_experiment_result.to_csv(path_search_result, sep = ',', index=False)

dict_criterio = {
    "class_termo" : {"type": "static",
               "value": {'class':['Termo']}},
    "total" : {"type": "static",
                  "value": {"count_index_total": {"$gte": 1} }},
    "area" : {"type": "static",
                   "value": {"count_index_area": {"$gte": 1} }},
    "theme" : {"type": "static",
                   "value": {"count_index_theme": {"$gte": 1} }},
    "subtheme" : {"type": "static",
                   "value": {"count_index_subtheme": {"$gte": 1} }},
    "extra" : {"type": "static",
                   "value": {"count_index_extra": {"$gte": 1} }},
    "total_gte_5" : {"type": "static",
                  "value": {"count_index_total": {"$gte": 5} }},
}

dict_idcg_relevance_fixed = generate_dict_idcg(15)

# print('dict_idcg_relevance_fixed', dict_idcg_relevance_fixed)


