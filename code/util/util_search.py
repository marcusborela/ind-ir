
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

MAIOR_INTEIRO = sys.maxsize


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



def calcular_rank1_um_documento (doctos_retornados, id_docto):
    """
    doctos_retornados: lista de documentos retornados na pesquisa (tem id como propriedade)
    id_docto: id do documento relevante
    """

    if len(doctos_retornados)>0:
        if isinstance(doctos_retornados[0], dict): #doctos_retornados[0]
            lista_id_docto_retornados = [int(docto['id']) for docto in doctos_retornados]
        else: # haystack.schema.Document
            lista_id_docto_retornados = [int(docto.id) for docto in doctos_retornados]
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



def buscar_documentos_para_experimento(parm_experimento, dados_termo):
    """
    Busca documentos conforme parâmetros
    """
    tamanho_limite = 1024

    if parm_experimento['criterio'] is not None and parm_experimento['tipo_retriever'] == 'tfidf':
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
    doctos_retornados = parm_experimento['pipe'].run(query=texto_pesquisa, params=parametro_busca)

    return doctos_retornados



def avalia_pipeline_experimento(parm_df,  parm_experimento,
                    parm_limite:int=MAIOR_INTEIRO,
                    parm_se_imprime:bool=False):
    """
    Cada experimento tem:
        {'retriever': {'bm25':pipeline_bm25_sem_preproc}, 'base':'vce_relevante_sem_preproc', 'criterio'= uma chave em dict_criterio},
    Em 9/5/2023
       mudando search_top_k para: topk_retriever e topk_ranker
    """

    qtd_dado = min(parm_df.shape[0],parm_limite)
    fracao_aleatoriedade = qtd_dado/len(parm_df)
    # print(f"fracao_aleatoriedade: {fracao_aleatoriedade}")
    df = parm_df.sample(frac=fracao_aleatoriedade, random_state=123).reset_index(drop=True)
    resultados = []
    tempo_inicio = time.time()
    soma_rank1 = 0
    qtd_rank1 = 0
    for cnt, row in tqdm(df.iterrows(), mininterval=10, total=qtd_dado):
        resultado_termo = {}
        resultado_termo['cod_termo_vce'] = row['COD_TERMO_VCE']
        resultado_termo['seq_treino'] = row['SEQ_TREINO']

        tempo_inicio_termo = time.time()
        doctos_retornados = buscar_documentos_para_experimento(parm_experimento=parm_experimento,  dados_termo=row)
        resultado_termo['qtd_retornado'] = len(doctos_retornados['documents'])
        if len(doctos_retornados) == 0:
            print(f"\nNÃO encontrou documentos em experimento {parm_experimento}\nCom parâmetros: {doctos_retornados['params']}")
            print(f"Com query: {doctos_retornados['query']}")
            print(f"Com metadados: Fonte: {row['FONTE']} Sistema: {row['SISTEMA']}")
        #else:
        #    print(f"\nNÃO encontrou documentos em experimento {parm_experimento}\nCom parâmetros: {doctos_retornados['params']}")
        #    print(f"Com query: {doctos_retornados['query']}")
        #    print(f"Com metadados: {row}")

        resultado_termo['tempo'] = round(time.time() - tempo_inicio_termo, 3)
        resultado_termo['rank1'] = calcular_rank1_um_documento(doctos_retornados['documents'], int(row['COD_TERMO_VCE']))
        if resultado_termo['rank1'] is not None:
            soma_rank1 += resultado_termo['rank1']
            qtd_rank1 += 1
        resultados.append(resultado_termo)
        if cnt >= qtd_dado - 1:
            break

    tempo_total = time.time() - tempo_inicio
    dict_retorno = {}
    dict_retorno['momento'] = time.strftime('%Y-%b-%d %H:%M:%S')
    dict_retorno['tipo_retriever'] = parm_experimento['tipo_retriever']
    dict_retorno['tipo_amostra'] = parm_experimento['tipo_amostra']
    if 'nome_modelo_retriever' in parm_experimento:
        dict_retorno['nome_modelo_retriever'] = parm_experimento['nome_modelo_retriever']
    else:
        dict_retorno['nome_modelo_retriever'] = ""
    if 'nome_modelo_ranker' in parm_experimento:
        dict_retorno['nome_modelo_ranker'] = parm_experimento['nome_modelo_ranker']
    else:
        dict_retorno['nome_modelo_ranker'] = ""
    dict_retorno['tipo_base'] = parm_experimento['tipo_base']
    dict_retorno['qtd_dado'] = qtd_dado
    dict_retorno['topk_retriever'] = parm_experimento['topk_retriever']
    dict_retorno['topk_retriever'] = parm_experimento['topk_retriever']
    if 'topk_ranker' in parm_experimento:
        dict_retorno['topk_ranker'] = parm_experimento['topk_ranker']
    else:
        dict_retorno['topk_ranker'] = ''

    if qtd_rank1 > 0:
        dict_retorno['rank1_mean'] = round(soma_rank1 / qtd_rank1)
    else:
        dict_retorno['rank1_mean'] = None
    dict_retorno['qtd_encontrado'] = qtd_rank1
    dict_retorno['qtd_nao_encontrado'] = qtd_dado - qtd_rank1
    dict_retorno['percent_nao_encontrado'] = round(100*dict_retorno['qtd_nao_encontrado']/qtd_dado)
    dict_retorno['tempo_medio_texto'] = round(tempo_total/qtd_dado,3)
    if parm_experimento['criterio'] is None:
        dict_retorno['criterio'] = ""
    else:
        dict_retorno['criterio'] = parm_experimento['criterio']
    dict_retorno['metrica_termos'] = resultados

    if parm_se_imprime:
        for conjunto in dict_retorno:
            for metric in dict_retorno[conjunto]:
                if isinstance(dict_retorno[conjunto][metric], str) or isinstance(dict_retorno[conjunto][metric], int):
                    print(f"{conjunto:>8}: {metric:>17}: {dict_retorno[conjunto][metric]}")
                else:
                    print(f"{conjunto:>8}: {metric:>17}: {dict_retorno[conjunto][metric]:.5f}")

    return dict_retorno



def add_experiment_result(parm_list_result, parm_path_experiment, parm_path_experiment_result):
    # criar um dataframe pandas a partir do dicionário
    df_experiment = pd.DataFrame(parm_list_result)

    # Inicialize o dataframe vazio para a saída
    df_experiment_result = pd.DataFrame()

    # Itere sobre cada linha da coluna METRICA_TERMOS e adicione ao dataframe de saída
    for _, row in df_experiment.iterrows():
        momento = row['momento']
        metricas = row['metrica_termos']
        for metrica in metricas:
            metrica['momento'] = momento
            df_experiment_result = df_experiment_result.append(metrica, ignore_index=True)
    del df_avaliacao['metrica_termos']
    df_experiment_result['cod_termo_vce'] = df_experiment_result['cod_termo_vce'].astype(int)
    df_experiment_result['seq_treino'] = df_experiment_result['seq_treino'].astype(int)
    df_experiment_result['qtd_retornado'] = df_experiment_result['qtd_retornado'].astype(int)
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
