
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

from util import util_pipeline

sys.path.append('../..')
sys.path


GREATEST_INTEGER = sys.maxsize

mapping_expansor_criteria = {
    'none': '----',
    'join_30_ptt5_base': 'base',
    'join_60_ptt5_indir_400': 'indir',
    'join_10_30_syn_or_rel_ptt5_indir_400': 'indir_extra',
}

mapping_ranker = {
    'none': '----',
    'PTT5_INDIR_400':'indir',
    'PTT5_INDIR_400_INV':'indiri',
    'PTT5_BASE':'base',
    'PTT5_BASE_INV':'basei'
}



def retorna_num_tokens(parm_texto:str, parm_tokenizador:AutoTokenizer):
    return len(parm_tokenizador.tokenize(parm_texto))

# Function to invert keys and values keeping a list of keys for each value
def invert_dict_with_lists(d):
    inverted_dict = defaultdict(list)
    for k, v in d.items():
        inverted_dict[v].append(k)
    return dict(inverted_dict)

def return_experiment(parm_dataset):
    path_search_experiment =  f'../data/search/{parm_dataset}/search_experiment_{parm_dataset}.csv'
    df_experiment = pd.read_csv(path_search_experiment)
    # consolidate data of queries
    if parm_dataset == 'juris_tcu_index':
        df_experiment['RANKER_TYPE'] = df_experiment.apply(find_ranker_type, axis=1)
    else:
        df_experiment['RANKER_TYPE'] = df_experiment['RANKER_TYPE'].fillna('none')
        df_experiment['EXPANSION_QUERY_COUNT'] = df_experiment['COLUMN_NAME'].apply(lambda x: x if x != 'TEXT' else '0').astype(int)

        df_experiment['EXPD_VAL'] = df_experiment.apply(define_expansion_doc_value, axis=1)

        # initial experiments did not have this information
        #df_experiment.loc[df_experiment['COLUMN_NAME'] == 'TEXT', 'EXPANSOR_CRITERIA'] = df_experiment.loc[df_experiment['COLUMN_NAME'] == 'TEXT', 'EXPANSOR_CRITERIA'].fillna("none")
        #df_experiment.loc[df_experiment['COLUMN_NAME'] != 'TEXT', 'EXPANSOR_CRITERIA'] = df_experiment.loc[df_experiment['COLUMN_NAME'] != 'TEXT', 'EXPANSOR_CRITERIA'].fillna("join_30_minilm_indir")

        df_experiment['RANKER_TYPE'] = df_experiment['RANKER_TYPE'].replace(mapping_ranker)
        df_experiment = df_experiment.rename(columns=lambda x: x.replace('_MEAN', '') if x.endswith('_MEAN') else x)
        df_experiment = df_experiment.rename(columns=lambda x: x.replace('_TYPE', '') if x.endswith('_TYPE') else x)
        df_experiment['EXPD_TYPE'] = df_experiment.apply(define_expansion_doc_type, axis=1)
        df_experiment['EXPQ_TYPE'] = df_experiment['EXPANSOR_CRITERIA'].replace(mapping_expansor_criteria)
        df_experiment.rename(columns={'EXPANSION_QUERY_COUNT':'EXPQ_CNT'},inplace=True)
        del df_experiment['EXPANSOR_CRITERIA']
        del df_experiment['INDEX_NAME']
        df_experiment['EXPQ_CNT'] = df_experiment['EXPQ_CNT'].astype(str).replace('0', '----')
        # Verificar as colunas com valores NaN
        columns_with_nan = df_experiment.columns[df_experiment.isna().any()].tolist()
        assert ['RETRIEVER_MODEL_NAME', 'EXPQ_TYPE'] == columns_with_nan, f"Review treatment of columns_with_nan. Expected: ['RETRIEVER_MODEL_NAME', 'EXPQ_TYPE'], found {columns_with_nan}"
        df_experiment = df_experiment.fillna("----")

        # adjust columns order by category

        list_expq_cnt_order = ['----', '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10']
        df_experiment['EXPQ_CNT'] = df_experiment['EXPQ_CNT'].astype(pd.CategoricalDtype(categories=list_expq_cnt_order, ordered=True))
        list_expq_type_order = ['----', 'base', 'indir', 'indir_extra']
        df_experiment['EXPQ_TYPE'] = df_experiment['EXPQ_TYPE'].astype(pd.CategoricalDtype(categories=list_expq_type_order, ordered=True))
        list_expd_val_order = ['----', 'term', '+syn', '+rel', '+syn+rel']
        df_experiment['EXPD_VAL'] = df_experiment['EXPD_VAL'].astype(pd.CategoricalDtype(categories=list_expd_val_order, ordered=True))
        list_expd_type_order = ['----', 'user', 'indir-1', 'indir-3', 'indir-5' ]
        df_experiment['EXPD_TYPE'] = df_experiment['EXPD_TYPE'].astype(pd.CategoricalDtype(categories=list_expd_type_order, ordered=True))
        list_retriever_order = [  'bm25', 'sts', 'join']
        df_experiment['RETRIEVER'] = df_experiment['RETRIEVER'].astype(pd.CategoricalDtype(categories=list_retriever_order, ordered=True))
        list_ranker_order = ['----', 'indir', 'indiri', 'base', 'basei']
        df_experiment['RANKER'] = df_experiment['RANKER'].astype(pd.CategoricalDtype(categories=list_ranker_order, ordered=True))




    return df_experiment

def define_expansion_doc_value(row):
    index_name = row['INDEX_NAME']

    if 'tcu_term_exp' in index_name:
        return 'term'
    elif 'tcu_synonym_exp' in index_name:
        return '+syn'
    elif 'tcu_related_term_exp' in index_name:
        return '+rel'
    elif 'tcu_synonym_related_term_exp' in index_name:
        return '+syn+rel'
    elif index_name == 'indir_juris_tcu':
        return '----'
    else:
        raise Exception(f"Expansion_doc_value unknown for index name: {index_name} ")

def define_expansion_doc_type(row):
    index_name = row['INDEX_NAME']

    if 'exp_1_ptt5_indir' in index_name:
        return 'indir-1'
    elif 'exp_3_ptt5_indir' in index_name:
        return 'indir-3'
    elif 'exp_5_ptt5_indir' in index_name:
        return 'indir-5'
    elif 'exp_user' in index_name:
        return 'user'
    elif index_name == 'indir_juris_tcu':
        return '----'
    else:
        raise Exception(f"Expansion_doc_criteria unknown for index name: {index_name} ")



def find_ranker_type(row):
    ranker_model_name = row['RANKER_MODEL_NAME']
    if pd.isnull(ranker_model_name) or ranker_model_name == "":
        return 'none'
    elif ranker_model_name in dict_ranker_dado_nome:
        return dict_ranker_dado_nome[ranker_model_name]
    else:
        print(f"Ranker type unknown {ranker_model_name} ")
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
    return df_result

def consolidate_result(parm_dataset):

    path_search_experiment =  f'../data/search/{parm_dataset}/search_experiment_{parm_dataset}.csv'
    path_search_result =  f'../data/search/{parm_dataset}/search_experiment_result_{parm_dataset}.csv'
    path_query = f'../data/{parm_dataset}/query.csv'
    path_qrel =  f'../data/{parm_dataset}/qrel.csv'
    path_search_result_consolidated = f'../data/search/{parm_dataset}/search_result_consolidated_{parm_dataset}.csv'


    # read experiment data
    df_experiment = return_experiment(parm_dataset)
    df_experiment_result = pd.read_csv(path_search_result)
    print(f'df_experiment.shape[0] {df_experiment.shape[0]}')
    print(f'df_experiment_result.shape[0] {df_experiment_result.shape[0]}')

    # merge experiment data
    df_experiment_result = df_experiment_result.merge(df_experiment, how='inner', on='TIME')
    print(f'df_experiment_result.shape {df_experiment_result.shape}')
    df_experiment_result.drop(df_experiment_result.filter(like='_MEAN').columns, axis=1, inplace=True)

    # read query/qrel data
    df_query = pd.read_csv(path_query)
    print(f'df_query.shape[0] {df_query.shape[0]}')

    df_qrel = pd.read_csv(path_qrel)
    print(f'df_qrel.shape[0] {df_qrel.shape[0]}')

    df_search_data = df_query.merge(df_qrel, how='left', left_on='ID', right_on='QUERY_ID').drop('QUERY_ID', axis=1)
    print(f'df_search_data.shape[0] {df_search_data.shape[0]}')
    # df_search_data.rename(columns={'TEXT': 'QUERY_TEXT', 'ID':'QUERY_ID'},inplace=True)

    # consolidate data of queries
    if parm_dataset == 'juris_tcu_index':
        df_new = df_search_data.groupby('ID').apply(lambda x: dict(zip(x['DOC_ID'], x['TYPE']))).reset_index(name='RELEVANCE_DICT_ID_DOC')
    else:
        df_new = df_search_data.groupby('ID').apply(lambda x: dict(zip(x['DOC_ID'], x['SCORE']))).reset_index(name='RELEVANCE_DICT_ID_DOC')
    df_new['RELEVANCE_DICT_TYPE'] = df_new['RELEVANCE_DICT_ID_DOC'].apply(invert_dict_with_lists)
    df_new = pd.merge(df_new, df_search_data.drop_duplicates('ID'), on='ID', how='left')

    df_new = df_new.add_prefix('QUERY_')
    # print('após add prefix', df_new.columns)


    # merge experiment data with queries searched
    df_experiment_result = df_experiment_result.merge(df_new, how='inner', left_on='QUERY_ID', right_on='QUERY_ID')# .drop('QUERY_ID', axis=1)


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


    # df_result['RANKER_TYPE'] = df_result['RANKER_MODEL_NAME'].apply(lambda x: 'none' if pd.isnull(x) else 'none' if x=="" else 'monot5' if isinstance(x, str) and 'mt5' in x.lower() else 'minilm' if isinstance(x, str) and 'minilm' in x.lower() else 'unknown')
    # if parm_dataset == 'juris_tcu_index':
    #     df_experiment_result['RANKER_TYPE'] = df_experiment_result.apply(find_ranker_type, axis=1)
    # else:
    #     df_experiment_result['RANKER_TYPE'] = df_experiment_result['RANKER_TYPE'].fillna('none')
    #     df_experiment_result['NUM_EXPANSAO'] = df_experiment_result['COLUMN_NAME'].apply(lambda x: x if x != 'TEXT' else '0').astype(int)




    # saving data
    columns_to_remove = [# about experiment data
                        # 'TIME', # 'COUNT_QUERY_RUN',
                         # about query/qrel data
                         'QUERY_TEXT',
                         'COUNT_QUERY_WITHOUT_RESULT','COUNT_QUERY_NOT_FOUND']
    #print([x for x in columns_to_remove if x not in df_experiment_result.columns])
    df_experiment_result.drop(columns_to_remove, axis=1, inplace=True)
    df_experiment_result.to_csv(path_search_result_consolidated, index=False)
    print(f"Generated file with {df_experiment_result.shape[0]} records")


def return_result_per_doc(parm_dataset):
    if parm_dataset == 'juris_tcu':
        raise Exception(f"The actual code only expect juris_tcu_index dataset! ")
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


dict_ranker_dado_nome = {v['model_name']: k for k, v in util_pipeline.dict_ranker.items()}