
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


def calculate_dict_idcg(dict_doc_relevant:dict, position:int):
    """
    calculate IDCG (Ideal Discounted Cumulative Gain) values.

    """
    sorted_values = sorted(dict_doc_relevant.values(), reverse=True)

    # print(sorted_values)

    idcg = 0

    for i, relevance in enumerate(sorted_values[:position]):
        if i >= position:
            raise ValueError('Logic error: more than position????')
        # relevance = 1 if docid in dict_doc_relevant else 0
        idcg += (2 ** relevance - 1) / math.log2(i + 2)

    return idcg

def calculate_precision_recall_query_result(list_id_doc_returned, dict_doc_relevant, position):
    """
    list_id_doc_returned: list of id of documents returned in search
    dict_doc_relevant: dict of relevance where id_doc is the key and type relevance (not used) value
    position: position at which to calculate precision and recall
    return precision, recall at position
    """
    if list_id_doc_returned is not None and len(list_id_doc_returned) > 0:
        relevant_count = 0
        for i, docid in enumerate(list_id_doc_returned[:position]):
            if i >= position:
                raise ValueError('Logic error: more than position???')

            if docid in dict_doc_relevant:
                if dict_doc_relevant[docid] >= 2:
                    relevant_count += 1

        precision = relevant_count / position
        recall = relevant_count / len(dict_doc_relevant)

        return precision, recall
    else:
        return 0, 0


def calculate_ndcg_query_result (list_id_doc_returned,
                                 dict_doc_relevant:dict,
                                 position:int)->float:
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
            # relevance = 1 if docid in dict_doc_relevant else 0
            if docid in dict_doc_relevant:
                relevance = dict_doc_relevant[docid]
            else:
                relevance = 0
            dcg += (2 ** relevance - 1) / math.log2(i + 2)

        # if len(dict_doc_relevant) not in dict_idcg_relevance_fixed:
        #     raise Exception(f"len(dict_doc_relevant) {len(dict_doc_relevant)} not in dict_idcg_relevance_fixed {dict_idcg_relevance_fixed}")
        # calculate ndcg for the query (Normalized Discounted Cumulative Gain)
        ndcg = dcg / calculate_dict_idcg(dict_doc_relevant, position)

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
        for doc_id in dict_doc_relevant:
            if dict_doc_relevant[doc_id] >= 2:
                if doc_id in list_id_doc_returned:
                    list_rank.append(1 + list_id_doc_returned.index(doc_id))  # 1st position of doc_id in the list
        return list_rank
    else:
        return None

def build_search_parameter(parm_experiment, query_data):
    dict_param_busca = {}
    if  'join' in parm_experiment['PIPE']['RETRIEVER_TYPE']:
        dict_param_busca = {"Bm25Retriever":{"top_k": np.int(parm_experiment['TOPK_RETRIEVER']/2)},
                            "StsRetriever": {"top_k": np.int(parm_experiment['TOPK_RETRIEVER']/2)}}
    else:
        dict_param_busca = {"Retriever":{"top_k": parm_experiment['TOPK_RETRIEVER']}}
    if 'TOPK_RANKER' in parm_experiment and parm_experiment['TOPK_RANKER'] > 0:
        dict_param_busca.update({"Ranker":{"top_k": parm_experiment['TOPK_RANKER']}})
    return dict_param_busca

def search_docto_for_experiment(parm_experiment, query_data):
    """
    Busca documentos conforme parâmetros
    """
    limit_size_text_first_stage = 1024
    search_parameter = build_search_parameter(parm_experiment, query_data)
    # print(f"search_parameter: {search_parameter} ")
    search_text =  query_data[parm_experiment['COLUMN_NAME']][:limit_size_text_first_stage]
    docto_found = parm_experiment['PIPE']['PIPE_OBJECT'].run(query=search_text, params=search_parameter)
    if 'documents' in docto_found: # search with pipe
        if len(docto_found['documents']) > 0:
            list_id_doc_returned = [docto.meta['id'] for docto in docto_found['documents']]
            return list_id_doc_returned #, search_parameter
    elif len(docto_found) > 0: # search with retriever
        list_id_doc_returned = [docto.meta['id'] for docto in docto_found]
        return list_id_doc_returned # , search_parameter
    else:
        return None

def experiment_run(parm_df,
                   parm_experiment,
                   parm_limit_query:int=GREATEST_INTEGER,
                   parm_print:bool=False):
    """
    Consider run as search for all queries
    """

    # param validation
    list_experiment_keys_expected = ['INDEX_NAME','TOPK_RETRIEVER', 'TOPK_RANKER', 'PIPE', 'COLUMN_NAME', 'EXPANSOR_CRITERIA']
    list_pipe_keys_expected = ['PIPE_OBJECT', 'RETRIEVER_TYPE', 'RETRIEVER_MODEL_NAME', 'RANKER_TYPE']

    # Verificar se todas as chaves de list_keys estão em parm_experiment
    if set(list_experiment_keys_expected) != set(parm_experiment.keys()):
        raise Exception(f"Invalid keys in parm_experiment {parm_experiment.keys()}. Expected: {list_experiment_keys_expected}")

    # Verificar se todas as chaves de list_keys estão em parm_experiment
    if set(list_pipe_keys_expected) != set(parm_experiment['PIPE'].keys()):
        raise Exception(f"Invalid keys in parm_experiment['PIPE'] {parm_experiment['PIPE'].keys()}. Expected: {list_pipe_keys_expected}")

    if parm_limit_query<parm_df.shape[0]:
        randomness_fraction = parm_limit_query/len(parm_df)
        df = parm_df.sample(frac=randomness_fraction, random_state=123).reset_index(drop=True)
        print(f"Experimento envolverá {df.shape[0]} registros com randomness_fraction: {randomness_fraction}")
    else:
        df = parm_df
        print(f"Experimento envolverá {df.shape[0]} registros, toda a base de dados")

    count_query_run = min(df.shape[0],parm_limit_query)

    result_search_all_query = []
    total_rank1 = 0 #
    total_ndcg5 = total_ndcg10 = total_ndcg15 = total_ndcg20 = 0
    total_precision_50 = total_precision_100 = total_recall_50 = total_recall_100 = 0
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
        result_search_one_query['QUERY_ID'] = row_query['ID']
        time_start_search_query = time.time()
        list_id_doc_returned = search_docto_for_experiment(parm_experiment=parm_experiment, query_data=row_query)
        result_search_one_query['TIME_SPENT'] = round((time.time() - time_start_search_query),4)
        if list_id_doc_returned is None or len(list_id_doc_returned) == 0:
            print(f"\nDocuments not found in experiment {parm_experiment} With query: {row_query['ID']}")
            result_search_one_query['RANK1'] = 0
            result_search_one_query['NDCG@5'] = 0
            result_search_one_query['NDCG@10'] = 0
            result_search_one_query['NDCG@15'] = 0
            result_search_one_query['NDCG@20'] = 0
            result_search_one_query['PRECISION@50'] = 0
            result_search_one_query['RECALL@50'] = 0
            result_search_one_query['PRECISION@100'] = 0
            result_search_one_query['RECALL@100'] = 0
            result_search_one_query['COUNT_DOCTO_FOUND'] = 0
            result_search_one_query['COUNT_DOCTO_RELEVANT'] = 0
            result_search_one_query['LIST_RANK'] = ""
            result_search_one_query['LIST_DOCTO_RETURNED'] = ""
            total_without_result += 1
        else:
            dict_ground_truth = row_query['RELEVANCE_DICT']
            if len(dict_ground_truth) == 0:
                raise Exception(f"Not expected query {row_query['ID']} without ground_truth")
            else:
                list_rank = calculate_list_rank_query_result(list_id_doc_returned, dict_ground_truth)
                if list_rank is None or len(list_rank) == 0:
                    result_search_one_query['RANK1'] = 0
                    result_search_one_query['NDCG@5'] = 0
                    result_search_one_query['NDCG@10'] = 0
                    result_search_one_query['NDCG@15'] = 0
                    result_search_one_query['NDCG@20'] = 0
                    result_search_one_query['PRECISION@50'] = 0
                    result_search_one_query['RECALL@50'] = 0
                    result_search_one_query['PRECISION@100'] = 0
                    result_search_one_query['RECALL@100'] = 0
                    result_search_one_query['COUNT_DOCTO_FOUND'] = 0
                    result_search_one_query['COUNT_DOCTO_RELEVANT'] = 0
                    result_search_one_query['LIST_RANK'] = ""
                    total_not_found += 1
                else:
                    result_search_one_query['NDCG@5'] = round(100*calculate_ndcg_query_result(list_id_doc_returned, dict_ground_truth, 5),2)
                    result_search_one_query['NDCG@10'] = round(100*calculate_ndcg_query_result(list_id_doc_returned, dict_ground_truth, 10),2)
                    result_search_one_query['NDCG@15'] = round(100*calculate_ndcg_query_result(list_id_doc_returned, dict_ground_truth, 15),2)
                    result_search_one_query['NDCG@20'] = round(100*calculate_ndcg_query_result(list_id_doc_returned, dict_ground_truth, 20),2)
                    precision, recall = calculate_precision_recall_query_result(list_id_doc_returned, dict_ground_truth, 50)
                    result_search_one_query['PRECISION@50'] = round(100*precision,3)
                    result_search_one_query['RECALL@50'] = round(100*recall,3)
                    precision, recall = calculate_precision_recall_query_result(list_id_doc_returned, dict_ground_truth, 100)
                    result_search_one_query['PRECISION@100'] = round(100*precision,3)
                    result_search_one_query['RECALL@100'] = round(100*recall,3)
                    result_search_one_query['COUNT_DOCTO_FOUND'] = len(list_id_doc_returned) if list_id_doc_returned is not None else 0
                    result_search_one_query['COUNT_DOCTO_RELEVANT'] = sum(1 for value in dict_ground_truth.values() if value >= 2)
                    result_search_one_query['GROUND_TRUTH'] = str(dict_ground_truth)
                    result_search_one_query['RANK1'] = min(list_rank)
                    result_search_one_query['LIST_RANK'] = str(list_rank)
                    total_found += 1
            result_search_one_query['LIST_DOCTO_RETURNED'] = str(list_id_doc_returned)
        # convertendo campos do tipo lista em string

        total_rank1 += result_search_one_query['RANK1']
        total_ndcg5 += result_search_one_query['NDCG@5']
        total_ndcg10 += result_search_one_query['NDCG@10']
        total_ndcg15 += result_search_one_query['NDCG@15']
        total_ndcg20 += result_search_one_query['NDCG@20']
        total_precision_50 += result_search_one_query['PRECISION@50']
        total_precision_100 += result_search_one_query['PRECISION@100']
        total_recall_50 += result_search_one_query['RECALL@50']
        total_recall_100 += result_search_one_query['RECALL@100']
        result_search_all_query.append(result_search_one_query)


    total_time = time.time() - time_start_search_run
    result_search_run = {}
    result_search_run['TIME'] = time.strftime('%Y-%b-%d %H:%M:%S')
    result_search_run['INDEX_NAME'] = parm_experiment['INDEX_NAME']
    result_search_run['COLUMN_NAME'] = parm_experiment['COLUMN_NAME']
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
    result_search_run['NDCG@5_MEAN'] = round(total_ndcg5/count_query_run,3)
    result_search_run['NDCG@10_MEAN'] = round(total_ndcg10/count_query_run,3)
    result_search_run['NDCG@15_MEAN'] = round(total_ndcg15/count_query_run,3)
    result_search_run['NDCG@20_MEAN'] = round(total_ndcg20/count_query_run,3)
    result_search_run['PRECISION@50_MEAN'] = round(total_precision_50/count_query_run,3)
    result_search_run['PRECISION@100_MEAN'] = round(total_precision_100/count_query_run,3)
    result_search_run['RECALL@50_MEAN'] = round(total_recall_50/count_query_run,3)
    result_search_run['RECALL@100_MEAN'] = round(total_recall_100/count_query_run,3)
    result_search_run['TIME_SPENT_MEAN'] = round(total_time/count_query_run,3)
    result_search_run['RETRIEVER_MODEL_NAME'] = parm_experiment['PIPE']['RETRIEVER_MODEL_NAME']
    result_search_run['RANKER_TYPE'] = parm_experiment['PIPE']['RANKER_TYPE']
    result_search_run['RESULT_QUERY'] = result_search_all_query

    if parm_print: # print results
        for key in result_search_run.keys():
            if key not in ['TIME','INDEX_NAME','RETRIEVER_TYPE','TOPK_RETRIEVER',\
                           'RETRIEVER_MODEL_NAME', 'RANKER_TYPE', 'RESULT_QUERY', 'TOPK_RANKER']:
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
    df_experiment_result['QUERY_ID'] = df_experiment_result['QUERY_ID'].astype(int)
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





