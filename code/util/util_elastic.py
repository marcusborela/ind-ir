import requests
from elasticsearch import Elasticsearch
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever


def return_indexes(parm_index_begin_name='indir', parm_print:bool=False):
    es = Elasticsearch()
    resposta = requests.get('http://localhost:9200/_cat/indices?v')

    # Dividir a string em linhas
    lines = resposta.text.strip().split("\n")

    # Extrair o cabeçalho
    header = lines[0].split()

    # Criar um dicionário para armazenar os índices
    index_dict = {}

    # Iterar pelas linhas a partir da segunda
    for line in lines[1:]:
        # Dividir a linha em colunas
        columns = line.split()

        # Criar um dicionário para o índice
        index_entry = {}

        # Preencher o dicionário com os valores das colunas
        for i, column in enumerate(columns):
            index_entry[header[i]] = column

        # Verificar se o nome do índice começa com 'indir%'
        if index_entry['index'].startswith(parm_index_begin_name):
            # Adicionar o índice ao dicionário principal
            index_dict[index_entry['index']] = index_entry

    if parm_print:
        for index_name, index_data in index_dict.items():
            print(f"Index: {index_name}")
            print(index_data)
            print()
        else:
            print(f"There are no index with name {parm_index_begin_name}%")

    return index_dict

def return_index(parm_index_name:str,  parm_embedding_dim:int=1024):


    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='', password='',
        index=parm_index_name,
        similarity='dot_product',
        search_fields="content",
        content_field = "content",
        embedding_field = "embedding",
        embedding_dim = parm_embedding_dim,
        duplicate_documents='fail',
        return_embedding=False,
    )
    print(f"\nQtd de documentos {doc_store.get_document_count()}")
    print(f"\nQtd de embeddings {doc_store.get_embedding_count()}")
    print(f"\nDocumento.id=1: {doc_store.get_document_by_id('1')}")
    return doc_store


def create_index(parm_index_name:str, parm_data_carga_json:list,  parm_embedding_dim:int=1024):

    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='', password='',
        index=parm_index_name,
        similarity='dot_product',
        search_fields="content",
        content_field = "content",
        embedding_field = "embedding",
        embedding_dim = parm_embedding_dim,
        # excluded_meta_data=['embedding'], # that should not be returned
        duplicate_documents='fail',
        return_embedding=False,
    )

    print(f"\nbefore write")
    id_test =  parm_data_carga_json[0]['id']
    print(f"\nQtd de documentos {doc_store.get_document_count()}")
    print(f"\nQtd de embeddings {doc_store.get_embedding_count()}")
    print(f"\nDocumento.id= {id_test}: {doc_store.get_document_by_id(id_test)}")

    doc_store.write_documents(parm_data_carga_json)
    print(f"\nafter write")
    print(f"\nQtd de documentos {doc_store.get_document_count()}")
    print(f"\nQtd de embeddings {doc_store.get_embedding_count()}")
    print(f"\nDocumento.id= {id_test}: {doc_store.get_document_by_id(id_test)}")

    return doc_store

def delete_index(parm_index_name:str):
    es = Elasticsearch()
    return es.indices.delete(index=parm_index_name, ignore=[400, 404])

def update_index_embedding_sts(parm_index, parm_path_model:str):
    retriever_sts = EmbeddingRetriever(
        document_store=parm_index,
        embedding_model=parm_path_model,
        model_format="sentence_transformers",
        pooling_strategy = 'cls_token',
        progress_bar = False
    )

    return parm_index.update_embeddings(retriever=retriever_sts)

def copy_index(parm_index_name_orige:str, parm_index_name_dest:str):
    # Copiando os dados
    # https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-reindex.html
    comando = {
    "source": {
        "index": parm_index_name_orige
    },
    "dest": {
        "index": parm_index_name_dest
    }
    }
    resposta = requests.post('http://localhost:9200/_reindex', headers = {"Content-Type": "application/json"}, json=comando)
    return resposta.status_code, resposta._content