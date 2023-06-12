
"""
rotinas de cálculo de métrica
"""
import sys
import os
import time
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack import Pipeline
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import SentenceTransformersRankerLimit
from haystack.nodes import MonoT5RankerLimit
from haystack.document_stores import ElasticsearchDocumentStore

import logging
logging.getLogger("haystack").setLevel(logging.WARNING) #WARNING, INFO

GREATEST_INTEGER = sys.maxsize

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


def return_pipeline_bm25(parm_index):
    retriever_bm25 = BM25Retriever(document_store=parm_index,all_terms_must_match=False)
    return DocumentSearchPipeline(retriever_bm25)

def return_pipeline_sts(parm_index:ElasticsearchDocumentStore, parm_path_model:str):
    retriever_sts = EmbeddingRetriever(
        document_store=parm_index,
        embedding_model=parm_path_model,
        model_format="sentence_transformers",
        pooling_strategy = 'cls_token',
        progress_bar = False
    )
    return DocumentSearchPipeline(retriever_sts)

def return_pipeline_bm25_reranker(parm_index:ElasticsearchDocumentStore, parm_ranker_type:str, parm_path_model_ranker:str, parm_limit_query_size:int=350):
    pipe_bm25_ranker = Pipeline()
    pipe_bm25_ranker.add_node(component= BM25Retriever(document_store=parm_index,all_terms_must_match=False), name="Retriever", inputs=["Query"])
    if parm_ranker_type == 'MONOT5':
        pipe_bm25_ranker.add_node(component=MonoT5RankerLimit(model_name_or_path=parm_path_model_ranker, limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])  # "Retriever" é o nome do nó anterior
    elif parm_ranker_type == 'MINILM':
        pipe_bm25_ranker.add_node(component=SentenceTransformersRankerLimit(model_name_or_path=parm_path_model_ranker, limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])  # "Retriever" é o nome do nó anterior
    else:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}")
    return pipe_bm25_ranker

def return_pipeline_sts_reranker(parm_index:ElasticsearchDocumentStore, parm_ranker_type:str, parm_path_model_ranker:str,  parm_path_model_sts:str, parm_limit_query_size:int=350):
    pipe_sts_ranker = Pipeline()
    pipe_sts_ranker.add_node(component= EmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=parm_path_model_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="Retriever", inputs=["Query"])
    if parm_ranker_type == 'MONOT5':
        pipe_sts_ranker.add_node(component=MonoT5RankerLimit(model_name_or_path=parm_path_model_ranker, limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])  # "Retriever" é o nome do nó anterior
    elif parm_ranker_type == 'MINILM':
        pipe_sts_ranker.add_node(component=SentenceTransformersRankerLimit(model_name_or_path=parm_path_model_ranker, limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])  # "Retriever" é o nome do nó anterior
    else:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}")
    return pipe_sts_ranker

def detail_document_found(parm_doc_returned):
    if 'params' in parm_doc_returned:
        print(f"Parâmetros usados: {parm_doc_returned['params']}")
        print(f"Consulta: {parm_doc_returned['query']}")
        print(f"Qtd documentos retornados: {len(parm_doc_returned['documents'])}")
        print(f'Primeiro docto:\n{parm_doc_returned["documents"][0]}\n\nÚltimo ({len(parm_doc_returned["documents"])}):\n{parm_doc_returned["documents"][-1]}')

        print(f'Seguem os nomes dos termos recuperados em ordem de score')
        doctos_dict = {ndx:[docto.meta['name'],docto.score] for ndx, docto in enumerate(parm_doc_returned['documents'])}
        for key, value in doctos_dict.items():
            print(key, ":", value)
    else: # retorno de reranking traz lista com documentos
        for docto in parm_doc_returned:
            print(docto.id, docto.score, docto.meta['name'])

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


def build_search_parameter(parm_experiment, query_data):
    dict_param_busca = {}
    if parm_experiment['CRITERIA'] is None:
        dict_param_busca = {"Retriever":{"top_k": parm_experiment['TOPK_RETRIEVER']}}
    else:
        criteria_type = dict_criterio[parm_experiment['CRITERIA']]['type']
        criteria_value = dict_criterio[parm_experiment['CRITERIA']]['value']
        if criteria_type == "static_filter":
            dict_param_busca = {"Retriever":{"top_k": parm_experiment['TOPK_RETRIEVER'], "filters": eval(criteria_value)}}
        elif criteria_type == "dynamic_filter" :
            dict_param_busca = {"Retriever":{"top_k": parm_experiment['TOPK_RETRIEVER'], "filters": criteria_value(query_data, parm_experiment['CRITERIA'])}}
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
    if 'documents' in docto_found: # search with ranker
        if len(docto_found['documents']) > 0:
            if isinstance(docto_found['documents'][0], dict):
                list_id_doc_returned = [int(docto['id']) for docto in docto_found['documents']]
            else:
                list_id_doc_returned = [int(docto.id) for docto in docto_found['documents']]
            return list_id_doc_returned #, search_parameter
    elif len(docto_found) > 0: # search without ranker
        if isinstance(docto_found[0], dict):
            list_id_doc_returned = [int(docto['id']) for docto in docto_found]
        else:
            list_id_doc_returned = [int(docto.id) for docto in docto_found]
        return list_id_doc_returned # , search_parameter

    else:
        return None

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

    count_query_run = min(parm_df.shape[0],parm_limit_query)
    randomness_fraction = count_query_run/len(parm_df)
    # print(f"randomness_fraction: {randomness_fraction}")
    df = parm_df.sample(frac=randomness_fraction, random_state=123).reset_index(drop=True)
    result_search_all_query = []
    total_rank1 = 0 #
    total_ndcg = 0
    cnt_rank1 = 0 # total de buscas em que se encantrou algum documento relevante na lista retornada
    time_start_search_run = time.time()
    for cnt, row_query in tqdm(df.iterrows(), mininterval=10, total=count_query_run):
        result_search_one_query = {}
        result_search_one_query['ID_QUERY'] = row_query['ID']
        result_search_one_query['COUNT_DOCTO_RELEVANT'] = len(row_query['RELEVANCE_DICT'])
        time_start_search_query = time.time()
        list_id_doc_returned = search_docto_for_experiment(parm_experiment=parm_experiment,  query_data=row_query)
        result_search_one_query['TIME_SPENT'] = round((time.time() - time_start_search_query),4)
        result_search_one_query['NDCG'] = round(100*calculate_ndcg_query_result(list_id_doc_returned, row_query['RELEVANCE_DICT'], parm_ndcg_position),2)
        result_search_one_query['COUNT_DOCTO_FOUND'] = len(list_id_doc_returned)
        list_rank = calculate_list_rank_query_result(list_id_doc_returned, row_query['RELEVANCE_DICT'])
        if len(list_rank) == 0:
            result_search_one_query['RANK1'] = 0
            result_search_one_query['LIST_RANK'] = ""
        else:
            result_search_one_query['RANK1'] = min(list_rank)
            total_rank1 += result_search_one_query['RANK1']
            result_search_one_query['LIST_RANK'] = str(list_rank)
            cnt_rank1 += 1
        # convertendo campos do tipo lista em string
        result_search_one_query['LIST_DOCTO_FOUND'] = str(list_id_doc_returned)
        if len(list_id_doc_returned) == 0:
            print(f"\nDocuments not found in experiment {parm_experiment}")
            # print(f"With parameters: {search_parameter}")
            print(f"With query: {row_query['TEXT']}")
        total_ndcg += result_search_one_query['NDCG']
        result_search_all_query.append(result_search_one_query)
        if cnt >= count_query_run - 1:
            break

    total_time = time.time() - time_start_search_run
    result_search_run = {}
    result_search_run['TIME'] = time.strftime('%Y-%b-%d %H:%M:%S')
    result_search_run['RETRIEVER_TYPE'] = parm_experiment['PIPE']['RETRIEVER_TYPE']
    result_search_run['COUNT_QUERY_RUN'] = count_query_run
    result_search_run['TOPK_RETRIEVER'] = parm_experiment['TOPK_RETRIEVER']
    result_search_run['TOPK_RANKER'] = parm_experiment['TOPK_RANKER']
    result_search_run['COUNT_QUERY_RUN'] = count_query_run
    result_search_run['RANK1_MEAN'] = round(total_rank1/cnt_rank1,3)
    result_search_run['NDCG_MEAN'] = round(total_ndcg/count_query_run,3)
    result_search_run['NDCG_LIMIT'] = parm_ndcg_position
    result_search_run['TIME_SPENT_MEAN'] = round(total_time/count_query_run,3)
    result_search_run['CRITERIA'] = parm_experiment['CRITERIA']
    result_search_run['RETRIEVER_MODEL_NAME'] = parm_experiment['PIPE']['RETRIEVER_MODEL_NAME']
    result_search_run['RANKER_MODEL_NAME'] = parm_experiment['PIPE']['RANKER_MODEL_NAME']
    result_search_run['RESULT_QUERY'] = result_search_all_query

    if parm_print: # print results
        for key in ['RANK1_MEAN','NDCG_MEAN','TIME_SPENT_MEAN']:
            print(f"{key:>8}: {result_search_run[key]}")

    return result_search_run



def add_experiment_result(parm_list_result, parm_path_experiment, parm_path_experiment_result):
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


    if not os.path.exists(parm_path_experiment):
        df_experiment.to_csv(parm_path_experiment, sep = ',', index=False)


        # Salvando coluna TIME no início
        # Lista de colunas existentes no DataFrame
        colunas = df_experiment_result.columns.tolist()
        # Move a coluna "TIME" para o início
        colunas.insert(0, colunas.pop(colunas.index('TIME')))
        # Reordena as colunas do DataFrame
        df_experiment_result = df_experiment_result[colunas]

        df_experiment_result.to_csv(parm_path_experiment_result, sep = ',', index=False)
    else: # concatenar a arquivo existente
        # Leitura do arquivo CSV existente
        df_experiment_save = pd.read_csv(parm_path_experiment)

        # Concatenação dos dataframes
        df_experiment = pd.concat([df_experiment_save, df_experiment], ignore_index=True)

        # Salvando o dataframe concatenado no arquivo CSV
        df_experiment.to_csv(parm_path_experiment, sep = ',', index=False)

        # Leitura do arquivo CSV existente
        df_experiment_result_save = pd.read_csv(parm_path_experiment_result)

        # Concatenação dos dataframes
        df_experiment_result = pd.concat([df_experiment_result_save, df_experiment_result], ignore_index=True)

        # Salvando o dataframe concatenado no arquivo CSV
        df_experiment_result.to_csv(parm_path_experiment_result, sep = ',', index=False)


def retorna_filtro_dinamico(parm_query_data, parm_criteria):
    if parm_criteria == 'class':
        valor = parm_query_data['CLASS']
        return {'class': valor}
    elif parm_criteria == 'total':
        return {"count_index_total": {"$gte": 1} }
    elif parm_criteria == 'area':
        return {"count_index_area": {"$gte": 1} }
    elif parm_criteria == 'theme':
        return {"count_index_theme": {"$gte": 1} }
    elif parm_criteria == 'subtheme':
        return {"count_index_subtheme": {"$gte": 1} }
    elif parm_criteria == 'total_gte_5':
        return {"count_index_total": {"$gte": 5} }

dict_criterio = {
    "class" : {"type": "dynamic_filter",
                   "value": retorna_filtro_dinamico},
    "total" : {"type": "dynamic_filter",
                  "value": retorna_filtro_dinamico},
    "area" : {"type": "dynamic_filter",
                   "value": retorna_filtro_dinamico},
    "theme" : {"type": "dynamic_filter",
                   "value": retorna_filtro_dinamico},
    "subtheme" : {"type": "dynamic_filter",
                   "value": retorna_filtro_dinamico},
    "total_gte_5" : {"type": "dynamic_filter",
                  "value": retorna_filtro_dinamico},
}

dict_idcg_relevance_fixed = generate_dict_idcg(15, 1)

# print('dict_idcg_relevance_fixed', dict_idcg_relevance_fixed)
