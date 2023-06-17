
"""
rotinas de cálculo de métrica
"""
# import sys
# import os
# import time
# import statistics
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import math
from haystack.nodes import EmbeddingRetriever, BM25Retriever, \
                            MonoT5RankerLimit, SentenceTransformersRankerLimit
from haystack.nodes import MultihopEmbeddingRetriever, JoinDocuments
from haystack import Pipeline
from haystack.pipelines import DocumentSearchPipeline
from haystack.document_stores import ElasticsearchDocumentStore

import logging
logging.getLogger("haystack").setLevel(logging.WARNING) #WARNING, INFO



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

def return_pipeline_sts_multihop(parm_index:ElasticsearchDocumentStore, parm_path_model:str):
    retriever_sts_multihop = MultihopEmbeddingRetriever(
        document_store=parm_index,
        embedding_model=parm_path_model,
        model_format="sentence_transformers",
        pooling_strategy = 'cls_token',
        progress_bar = False
    )
    return DocumentSearchPipeline(retriever_sts_multihop)

def return_pipeline_join(parm_index:ElasticsearchDocumentStore,
                                  parm_path_model_sts:str):
    # doc in https://docs.haystack.deepset.ai/reference/other-api#joinanswers__init__
    pipe_join = Pipeline()
    pipe_join.add_node(component= BM25Retriever(document_store=parm_index,
                                                       all_terms_must_match=False),
                              name="Bm25Retriever", inputs=["Query"])
    pipe_join.add_node(component= EmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=parm_path_model_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="StsRetriever", inputs=["Query"])
    pipe_join.add_node(component=JoinDocuments(join_mode="concatenate"),
                              name="JoinResults",
                              inputs=["Bm25Retriever", "StsRetriever"])

    return pipe_join

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

def return_pipeline_sts_multihop_reranker(parm_index:ElasticsearchDocumentStore, parm_ranker_type:str, parm_path_model_ranker:str,  parm_path_model_sts:str, parm_limit_query_size:int=350):
    pipe_sts_multihop_ranker = Pipeline()
    pipe_sts_multihop_ranker.add_node(component= MultihopEmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=parm_path_model_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="Retriever", inputs=["Query"])
    if parm_ranker_type == 'MONOT5':
        pipe_sts_multihop_ranker.add_node(component=MonoT5RankerLimit(model_name_or_path=parm_path_model_ranker, limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])  # "Retriever" é o nome do nó anterior
    elif parm_ranker_type == 'MINILM':
        pipe_sts_multihop_ranker.add_node(component=SentenceTransformersRankerLimit(model_name_or_path=parm_path_model_ranker, limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])  # "Retriever" é o nome do nó anterior
    else:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}")
    return pipe_sts_multihop_ranker

def return_pipeline_join_reranker(parm_index:ElasticsearchDocumentStore,
                                  parm_ranker_type:str,
                                  parm_path_model_ranker:str,
                                  parm_path_model_sts:str,
                                  parm_limit_query_size:int=350):
    pipe_join_ranker = Pipeline()
    pipe_join_ranker.add_node(component= BM25Retriever(document_store=parm_index,
                                                       all_terms_must_match=False),
                              name="Bm25Retriever", inputs=["Query"])
    pipe_join_ranker.add_node(component= EmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=parm_path_model_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="StsRetriever", inputs=["Query"])
    pipe_join_ranker.add_node(component=JoinDocuments(join_mode="concatenate"),
                              name="JoinResults",
                              inputs=["Bm25Retriever", "StsRetriever"])

    if parm_ranker_type == 'MONOT5':
        pipe_join_ranker.add_node(component=MonoT5RankerLimit(model_name_or_path=parm_path_model_ranker,
                                                              limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["JoinResults"])
    else:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}")
    return pipe_join_ranker

def return_pipeline_join(parm_index:ElasticsearchDocumentStore,
                                  parm_path_model_sts:str):
    pipe_join = Pipeline()
    pipe_join.add_node(component= BM25Retriever(document_store=parm_index,
                                                       all_terms_must_match=False),
                              name="Bm25Retriever", inputs=["Query"])
    pipe_join.add_node(component= EmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=parm_path_model_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="StsRetriever", inputs=["Query"])
    pipe_join.add_node(component=JoinDocuments(join_mode="concatenate"),
                              name="JoinResults",
                              inputs=["Bm25Retriever", "StsRetriever"])

    return pipe_join

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

