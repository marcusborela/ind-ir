
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

from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack import Pipeline
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import SentenceTransformersRankerLimit
from haystack.nodes import MonoT5RankerLimit
from haystack.document_stores import ElasticsearchDocumentStore

import logging
logging.getLogger("haystack").setLevel(logging.WARNING) #WARNING, INFO

MAIOR_INTEIRO = sys.maxsize


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

def calcular_rank1_um_documento (docto_found, id_docto):
    """
    docto_found: lista de documentos retornados na pesquisa (tem id como propriedade)
    id_docto: id do documento relevante
    """

    if len(docto_found)>0:
        if isinstance(docto_found[0], dict): #docto_found[0]
            lista_id_docto_retornados = [int(docto['id']) for docto in docto_found]
        else: # haystack.schema.Document
            lista_id_docto_retornados = [int(docto.id) for docto in docto_found]
        if id_docto in lista_id_docto_retornados:
            return 1 + lista_id_docto_retornados.index(id_docto) # 1a posição do id_docto na lista
        else:
            return None
    else:
        # print('não há documentos retornados para id_docto {id_docto}')
        return None

def retorna_parametro(parm_experimento, dados_termo):
    dict_param_busca = {}
    if parm_experimento['criterio'] is None:
        dict_param_busca = {"Retriever":{"top_k": parm_experimento['topk_retriever']}}
    else:
        criterio_tipo = dict_criterio[parm_experimento['criterio']]['tipo']
        criterio_valor = dict_criterio[parm_experimento['criterio']]['valor']
        if criterio_tipo == "filtro_estatico":
            dict_param_busca = {"Retriever":{"top_k": parm_experimento['topk_retriever'], "filters": eval(criterio_valor)}}
        elif criterio_tipo == "filtro_dinamico_campo" :
            dict_param_busca = {"Retriever":{"top_k": parm_experimento['topk_retriever'], "filters": criterio_valor(dados_termo, parm_experimento['criterio'])}}
    if 'topk_ranker' in parm_experimento:
        dict_param_busca.update({"Ranker":{"top_k": parm_experimento['topk_ranker']}})
    return dict_param_busca

def search_docto_for_experiment(parm_experimento, query_data):
    """
    Busca documentos conforme parâmetros
    """
    tamanho_limite = 1024

    if parm_experimento['criterio'] is not None and parm_experimento['RETRIEVER_TYPE'] == 'tfidf':
        raise Exception (f'Retriever tfidf não aceita critérios com filtros')

    # para evitar erro es.maxClauseCount is set to 1024
    if 'sem_preproc' in parm_experimento['tipo_base']:
        texto_pesquisa = dados_termo['TEXTO']
    elif 'com_preproc_sem_steem' in parm_experimento['tipo_base']:
        texto_pesquisa = dados_termo['TEXTO_PREPROC_SEM_STEEM']
    elif 'com_preproc_com_steem' in parm_experimento['tipo_base']:
        texto_pesquisa = dados_termo['TEXTO_PREPROC_COM_STEEM']
    else:
        raise Exception(f"parm_experimento['tipo_base'] {parm_experimento['tipo_base']} não tem mapeamento de coluna com o texto para a busca!")

    # texto_pesquisa = texto_pesquisa[:tamanho_limite]

    parametro_busca = retorna_parametro(parm_experimento, dados_termo)
    # print(f"parametro_busca: {parametro_busca} ")


    texto_pesquisa = texto_pesquisa[:tamanho_limite]
    docto_found = parm_experimento['pipe'].run(query=texto_pesquisa, params=parametro_busca)

    return docto_found

def experiment_run(parm_df,  parm_experiment,
                    parm_limit_query:int=MAIOR_INTEIRO,
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
    fracao_aleatoriedade = count_query_run/len(parm_df)
    # print(f"fracao_aleatoriedade: {fracao_aleatoriedade}")
    df = parm_df.sample(frac=fracao_aleatoriedade, random_state=123).reset_index(drop=True)
    result_search_all_query = []
    time_start_search_run = time.time()
    for cnt, row_query in tqdm(df.iterrows(), mininterval=10, total=count_query_run):
        result_search_one_query = {}
        result_search_one_query['ID_QUERY'] = row_query['ID']
        time_start_search_query = time.time()
        docto_found = search_docto_for_experiment(parm_experiment=parm_experiment,  query_data=row_query)
        result_search_one_query['COUNT_DOCTO_FOUND'] = len(docto_found['documents'])
        if len(docto_found) == 0:
            print(f"\nDocuments not found in experiment {parm_experiment}\nWith parameters: {docto_found['params']}")
            print(f"With query: {docto_found['query']}")

        result_search_one_query['TIME_SPENT'] = round(time.time() - time_start_search_query, 3)
        result_search_all_query.append(result_search_one_query)
        if cnt >= count_query_run - 1:
            break

    total_time = time.time() - time_start_search_run
    result_search_run = {}
    result_search_run['TIME'] = time.strftime('%Y-%b-%d %H:%M:%S')
    result_search_run['RETRIEVER_TYPE'] = parm_experiment['PIPE']['RETRIEVER_TYPE']
    result_search_run['RETRIEVER_MODEL_NAME'] = parm_experiment['PIPE']['RETRIEVER_MODEL_NAME']
    result_search_run['RANKER_MODEL_NAME'] = parm_experiment['PIPE']['RANKER_MODEL_NAME']
    result_search_run['COUNT_QUERY_RUN'] = count_query_run
    result_search_run['TOPK_RETRIEVER'] = parm_experiment['TOPK_RETRIEVER']
    result_search_run['TOPK_RANKER'] = parm_experiment['TOPK_RANKER']

    result_search_run['COUNT_QUERY_RUN'] = count_query_run
    result_search_run['TIME_SPENT_MEAN'] = round(total_time/count_query_run,3)
    result_search_run['CRITERIA'] = parm_experiment['CRITERIA']
    result_search_run['RESULT_QUERY'] = result_search_all_query

    if parm_print:
        for conjunto in result_search_run:
            for metric in result_search_run[conjunto]:
                if isinstance(result_search_run[conjunto][metric], str) or isinstance(result_search_run[conjunto][metric], int):
                    print(f"{conjunto:>8}: {metric:>17}: {result_search_run[conjunto][metric]}")
                else:
                    print(f"{conjunto:>8}: {metric:>17}: {result_search_run[conjunto][metric]:.5f}")

    return result_search_run



def add_experiment_result(parm_list_result, parm_path_experiment, parm_path_experiment_result):
    # criar um dataframe pandas a partir do dicionário
    df_experiment = pd.DataFrame(parm_list_result)

    # Inicialize o dataframe vazio para a saída
    df_experiment_result = pd.DataFrame()

    # Itere sobre cada linha da coluna METRICA_TERMOS e adicione ao dataframe de saída
    for _, row_query in df_experiment.iterrows():
        TIME = row_query['TIME']
        metricas = row_query['metrica_termos']
        for metrica in metricas:
            metrica['TIME'] = TIME
            df_experiment_result = df_experiment_result.append(metrica, ignore_index=True)
    del df_avaliacao['metrica_termos']
    df_experiment_result['cod_termo_vce'] = df_experiment_result['cod_termo_vce'].astype(int)
    df_experiment_result['seq_treino'] = df_experiment_result['seq_treino'].astype(int)
    df_experiment_result['COUNT_DOCTO_FOUND'] = df_experiment_result['COUNT_DOCTO_FOUND'].astype(int)
    # df_experiment_result['qtd_relevante_encontrado'] = df_experiment_result['qtd_relevante_encontrado'].astype(int)
    # df_experiment_result['qtd_relevante_topk'] = df_experiment_result['qtd_relevante_topk'].astype(int)

    if not os.path.exists(parm_path_experiment):
        df_experiment.to_csv(parm_path_experiment, sep = ',', index=False)
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


def retorna_filtro_dinamico(parm_query_data, parm_criterio):
    if parm_criterio == 'classe':
        valor = parm_query_data['CLASSE']
        return {'class': valor}
    elif parm_criterio == 'total':
        return {"count_index_total": {"$gte": 1} }
    elif parm_criterio == 'area':
        return {"count_index_area": {"$gte": 1} }
    elif parm_criterio == 'theme':
        return {"count_index_theme": {"$gte": 1} }
    elif parm_criterio == 'subtheme':
        return {"count_index_subtheme": {"$gte": 1} }
    elif parm_criterio == 'total_gte_5':
        return {"count_index_total": {"$gte": 5} }

dict_criterio = {
    "classe" : {"tipo": "filtro_dinamico_campo",
                   "valor": retorna_filtro_dinamico},
    "total" : {"tipo": "filtro_dinamico_campo",
                  "valor": retorna_filtro_dinamico},
    "area" : {"tipo": "filtro_dinamico_campo",
                   "valor": retorna_filtro_dinamico},
    "theme" : {"tipo": "filtro_dinamico_campo",
                   "valor": retorna_filtro_dinamico},
    "subtheme" : {"tipo": "filtro_dinamico_campo",
                   "valor": retorna_filtro_dinamico},
    "total_gte_5" : {"tipo": "filtro_dinamico_campo",
                  "valor": retorna_filtro_dinamico},
}
