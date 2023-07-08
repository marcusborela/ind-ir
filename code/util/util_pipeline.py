
"""
rotinas de cálculo de métrica
"""
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack.nodes import MonoT5RankerLimit, SentenceTransformersRankerLimit
from haystack.nodes import MultihopEmbeddingRetriever, JoinDocuments
from haystack import Pipeline
from haystack.pipelines import DocumentSearchPipeline
from haystack.document_stores import ElasticsearchDocumentStore

import logging
logging.getLogger("haystack").setLevel(logging.WARNING) #WARNING, INFO


def return_ranker(parm_ranker_type:str, parm_limit_query_size:int=350):
    global dict_ranker
    if parm_ranker_type not in dict_ranker:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}. Must be in {dict_ranker.keys()}")
    elif parm_limit_query_size not in (50, 100, 350):
        raise Exception (f"Invalid parm_limit_query_size {parm_limit_query_size} for singleton code!")
    else:
        if dict_ranker[parm_ranker_type]['limit_query_size'] is not None:
            if parm_limit_query_size != dict_ranker[parm_ranker_type]['limit_query_size']:
                raise Exception (f"Not expected parm_limit_query_size {parm_limit_query_size} for ranker built in singleton with size {dict_ranker[parm_ranker_type]['limit_query_size']} !")
        if dict_ranker[parm_ranker_type]['model'] is None:
            print(f"Loading {parm_ranker_type} with limit_query_size={parm_limit_query_size}")
            if dict_ranker[parm_ranker_type]['inference_type'] == 'SeqClassification':
                dict_ranker[parm_ranker_type]['model'] = SentenceTransformersRankerLimit(model_name_or_path=f"{PATH_MODELO}/{dict_ranker[parm_ranker_type]['model_name']}", limit_query_size=parm_limit_query_size)
            elif dict_ranker[parm_ranker_type]['inference_type'] == 'Seq2SeqLM':
                dict_ranker[parm_ranker_type]['model'] = MonoT5RankerLimit(model_name_or_path=f"{PATH_MODELO}/{dict_ranker[parm_ranker_type]['model_name']}", limit_query_size=parm_limit_query_size)
            else:
                raise Exception (f"Invalid model.inference_type {dict_ranker[parm_ranker_type]['inference_type']}. Must be in [SeqClassification, Seq2SeqLM ]")
            dict_ranker[parm_ranker_type]['limit_query_size'] = parm_limit_query_size
        return dict_ranker[parm_ranker_type]['model']


def return_ranker_name(parm_ranker_type:str)->str:
    global dict_ranker
    if parm_ranker_type not in dict_ranker:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}. Must be in {dict_ranker.keys()}")

    return dict_ranker[parm_ranker_type]['model_name']

def return_multihop_embedding_retriever(parm_index:ElasticsearchDocumentStore):
    return MultihopEmbeddingRetriever(
                    document_store=parm_index,
                    embedding_model=nome_caminho_modelo_sts,
                    model_format="sentence_transformers",
                    pooling_strategy = 'cls_token',
                    progress_bar = False
                )

def return_pipeline_bm25(parm_index:ElasticsearchDocumentStore):
    retriever_bm25 = BM25Retriever(document_store=parm_index,all_terms_must_match=False)
    return DocumentSearchPipeline(retriever_bm25)

def return_pipeline_sts(parm_index:ElasticsearchDocumentStore):
    retriever_sts = EmbeddingRetriever(
        document_store=parm_index,
        embedding_model=nome_caminho_modelo_sts,
        model_format="sentence_transformers",
        pooling_strategy = 'cls_token',
        progress_bar = False
    )
    return DocumentSearchPipeline(retriever_sts)

def return_pipeline_sts_multihop(parm_index:ElasticsearchDocumentStore):
    retriever_sts_multihop = return_multihop_embedding_retriever(parm_index)
    return DocumentSearchPipeline(retriever_sts_multihop)

def return_pipeline_join(parm_index:ElasticsearchDocumentStore):
    # doc in https://docs.haystack.deepset.ai/reference/other-api#joinanswers__init__
    pipe_join = Pipeline()
    pipe_join.add_node(component= BM25Retriever(document_store=parm_index,
                                                       all_terms_must_match=False),
                              name="Bm25Retriever", inputs=["Query"])
    pipe_join.add_node(component= EmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=nome_caminho_modelo_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="StsRetriever", inputs=["Query"])
    pipe_join.add_node(component=JoinDocuments(join_mode="concatenate"),
                              name="JoinResults",
                              inputs=["Bm25Retriever", "StsRetriever"])

    return pipe_join

def return_pipeline_bm25_reranker(parm_index:ElasticsearchDocumentStore, parm_ranker_type:str='MONOT5', parm_limit_query_size:int=350):
    pipe_bm25_ranker = Pipeline()
    pipe_bm25_ranker.add_node(component= BM25Retriever(document_store=parm_index,all_terms_must_match=False), name="Retriever", inputs=["Query"])

    if parm_ranker_type not in dict_ranker:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}. Must be in {dict_ranker.keys()}")
    else:
        pipe_bm25_ranker.add_node(component=return_ranker(parm_ranker_type, parm_limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])
    return pipe_bm25_ranker

def return_pipeline_sts_reranker(parm_index:ElasticsearchDocumentStore,
                                 parm_ranker_type:str='MONOT5',
                                 parm_limit_query_size:int=350):
    pipe_sts_ranker = Pipeline()
    pipe_sts_ranker.add_node(component= EmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=nome_caminho_modelo_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="Retriever", inputs=["Query"])
    if parm_ranker_type not in dict_ranker:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}. Must be in {dict_ranker.keys()}")
    else:
        pipe_sts_ranker.add_node(component=return_ranker(parm_ranker_type, parm_limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])

    return pipe_sts_ranker

def return_pipeline_sts_multihop_reranker(parm_index:ElasticsearchDocumentStore,
                                          parm_ranker_type:str='MONOT5',
                                          parm_limit_query_size:int=350):
    pipe_sts_multihop_ranker = Pipeline()
    pipe_sts_multihop_ranker.add_node(component= return_multihop_embedding_retriever(parm_index),
                             name="Retriever", inputs=["Query"])

    if parm_ranker_type not in dict_ranker:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}. Must be in {dict_ranker.keys()}")
    else:
        pipe_sts_multihop_ranker.add_node(component=return_ranker(parm_ranker_type, parm_limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["Retriever"])
    return pipe_sts_multihop_ranker

def return_pipeline_join_bm25_sts_reranker(parm_index:ElasticsearchDocumentStore,
                                  parm_ranker_type:str='MONOT5',
                                  parm_limit_query_size:int=350):
    pipe_join_ranker = Pipeline()
    pipe_join_ranker.add_node(component= BM25Retriever(document_store=parm_index,
                                                       all_terms_must_match=False),
                              name="Bm25Retriever", inputs=["Query"])
    pipe_join_ranker.add_node(component= EmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=nome_caminho_modelo_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="StsRetriever", inputs=["Query"])
    pipe_join_ranker.add_node(component=JoinDocuments(join_mode="concatenate"),
                              name="JoinResults",
                              inputs=["Bm25Retriever", "StsRetriever"])

    if parm_ranker_type not in dict_ranker:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}. Must be in {dict_ranker.keys()}")
    else:
        pipe_join_ranker.add_node(component=return_ranker(parm_ranker_type, parm_limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["JoinResults"])
    return pipe_join_ranker

def return_pipeline_join_bm25_sts(parm_index:ElasticsearchDocumentStore):
    pipe_join = Pipeline()
    pipe_join.add_node(component= BM25Retriever(document_store=parm_index,
                                                       all_terms_must_match=False),
                              name="Bm25Retriever", inputs=["Query"])
    pipe_join.add_node(component= EmbeddingRetriever(
                                                            document_store=parm_index,
                                                            embedding_model=nome_caminho_modelo_sts,
                                                            model_format="sentence_transformers",
                                                            pooling_strategy = 'cls_token',
                                                            progress_bar = False),
                             name="StsRetriever", inputs=["Query"])
    pipe_join.add_node(component=JoinDocuments(join_mode="concatenate"),
                              name="JoinResults",
                              inputs=["Bm25Retriever", "StsRetriever"])

    return pipe_join


def return_pipeline_join_bm25_sts_multihop_reranker(parm_index:ElasticsearchDocumentStore,
                                  parm_ranker_type:str='MONOT5',
                                  parm_limit_query_size:int=350):
    pipe_join_ranker = Pipeline()
    pipe_join_ranker.add_node(component= BM25Retriever(document_store=parm_index,
                                                       all_terms_must_match=False),
                              name="Bm25Retriever", inputs=["Query"])
    pipe_join_ranker.add_node(component= return_multihop_embedding_retriever(parm_index),
                             name="StsRetriever", inputs=["Query"])
    pipe_join_ranker.add_node(component=JoinDocuments(join_mode="concatenate"),
                              name="JoinResults",
                              inputs=["Bm25Retriever", "StsRetriever"])

    if parm_ranker_type not in dict_ranker:
        raise Exception (f"Invalid parm_ranker_type {parm_ranker_type}. Must be in {dict_ranker.keys()}")
    else:
        pipe_join_ranker.add_node(component=return_ranker(parm_ranker_type, parm_limit_query_size=parm_limit_query_size),
                                        name="Ranker", inputs=["JoinResults"])
    return pipe_join_ranker

def return_pipeline_join_bm25_sts_multihop(parm_index:ElasticsearchDocumentStore):
    pipe_join = Pipeline()
    pipe_join.add_node(component= BM25Retriever(document_store=parm_index,
                                                       all_terms_must_match=False),
                              name="Bm25Retriever", inputs=["Query"])
    pipe_join.add_node(component= return_multihop_embedding_retriever(parm_index),
                             name="StsRetriever", inputs=["Query"])
    pipe_join.add_node(component=JoinDocuments(join_mode="concatenate"),
                              name="JoinResults",
                              inputs=["Bm25Retriever", "StsRetriever"])

    return pipe_join


def detail_document_found(parm_doc_returned, parm_num_doc:int=10):
    if 'params' in parm_doc_returned:
        print(f"Parâmetros usados: {parm_doc_returned['params']}")
        print(f"Consulta: {parm_doc_returned['query']}")
        print(f"Qtd documentos retornados: {len(parm_doc_returned['documents'])}")
        if len(parm_doc_returned['documents']) > 0:
            # print(f'Primeiro docto:\n{parm_doc_returned["documents"][0]}\n\nÚltimo ({len(parm_doc_returned["documents"])}):\n{parm_doc_returned["documents"][-1]}')

            print(f'Seguem os nomes dos termos recuperados em ordem de score')
            if 'name' in parm_doc_returned['documents'][0].meta: # juris_tcu_index
                doctos_dict = {ndx:[docto.meta['name'],docto.id, docto.score] for ndx, docto in enumerate(parm_doc_returned['documents'])}
            else: # juris_tcu_index
                doctos_dict = {ndx:[docto.score, docto.id, docto.content] for ndx, docto in enumerate(parm_doc_returned['documents'])}
            for count, (key, value) in enumerate(doctos_dict.items()):
                if count > parm_num_doc:
                    break
                print(key, ":", value)
    else: # retorno de reranking traz lista com documentos
        if len(parm_doc_returned) > 0:
            for count, docto in enumerate(parm_doc_returned):
                if count > parm_num_doc:
                    break
                print(docto.id, docto.score, docto.meta['name'])



def print_pipe_image(parm_pipe):
    # Criar um arquivo temporário para salvar a imagem
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        # Salvar o pipe no arquivo temporário
        parm_pipe.draw(temp_file.name)

        # Exibir a imagem no Jupyter Notebook
        img = mpimg.imread(temp_file.name)
        plt.imshow(img)
        plt.axis('off')
        plt.show()



# more commands
# from pathlib import Path
# ? pip.get_config(return_defaults=parm_return_defaults)
# pipe_join_ranker_monot5_3b.save_to_yaml(Path("pipe_join_ranker_monot5_3b.yahml"), return_defaults = True)

PATH_MODELO = "/home/borela/fontes/relevar-busca/modelo"


nome_modelo_embedding_model_sts = "rufimelo/Legal-BERTimbau-sts-large-ma-v3"
nome_caminho_modelo_sts = "/home/borela/fontes/relevar-busca/modelo/" + nome_modelo_embedding_model_sts
assert os.path.exists(nome_caminho_modelo_sts), f"Path para {nome_caminho_modelo_sts} não existe!"


dict_ranker = {
    'MINILM'  : {'limit_query_size': None, 'model': None,
                 'inference_type': 'SeqClassification',
                 'model_name': 'unicamp-dl/mMiniLM-L6-v2-pt-v2'},
    'MT5_3B'  : {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/mt5-3B-mmarco-en-pt' },
    'PTT5_BASE': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2' },
    'PTT5_TRAINED_1400': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-1400' },
    'PTT5_TRAINED_5200': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-5200' },
    'PTT5_TRAINED_7600': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-7600' },
    'PTT5_TRAINED_LIM100_1700': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-lim100-1700' },
    'PTT5_TRAINED_LIM50_800': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-lim50-800' },
    'PTT5_TRAINED_LIM50_2200': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-lim50-2200' },
    'PTT5_INDIR_41': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-41-pcte' },
    'PTT5_INDIR_79': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-79-pcte' },
    'PTT5_INDIR_83': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-83-pcte' },
    'PTT5_INDIR_106': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-106-pcte' },
    'PTT5_INDIR_266': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-266-pcte' },
    'PTT5_INDIR_400': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-400-pcte' },
    'PTT5_INDIR_600': {'limit_query_size': None, 'model': None,
                 'inference_type': 'Seq2SeqLM',
                 'model_name': 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-600-pcte' },
    'MINILM_TRAINED_5500'  : {'limit_query_size': None, 'model': None,
                 'inference_type': 'SeqClassification',
                 'model_name': 'unicamp-dl/mMiniLM-L6-v2-pt-v2-5500'},
    'MINILM_TRAINED_19000'  : {'limit_query_size': None, 'model': None,
                 'inference_type': 'SeqClassification',
                 'model_name': 'unicamp-dl/mMiniLM-L6-v2-pt-v2-19000'},
    'MINILM_TRAINED_49200'  : {'limit_query_size': None, 'model': None,
                 'inference_type': 'SeqClassification',
                 'model_name': 'unicamp-dl/mMiniLM-L6-v2-pt-v2-49200'},
    'MINILM_INDIR_400' : {'limit_query_size': None, 'model': None,
                 'inference_type': 'SeqClassification',
                 'model_name': 'unicamp-dl/mMiniLM-L6-v2-pt-v2-indir-400-pcte'},

}

for ranker in dict_ranker:
    if 'TRAINED' not in ranker: # trained version has been deleted, just models in trainnig steps  to get to indir
        model_name = dict_ranker[ranker]['model_name']
        assert os.path.exists(f"{PATH_MODELO}/{model_name}"), f"Caminho de modelo {ranker} não existe: {PATH_MODELO}/{model_name}"

# ranker_monot5_3b = None
# ranker_limit_query_size_monot5_3b = None


# ranker_monot5_base = None
# ranker_limit_query_size_monot5_base = None

# dict_multihop_embedding_retriever = {'indir_juris_tcu': None, 'indir_juris_tcu_index':None}
