# IndIR - Datasets e outros arquivos preparados

## [JURIS_TCU](/data/juris_tcu/)
Base de enunciados da [Jurisprudência Selecionada do Tribunal de Contas da União](https://portal.tcu.gov.br/jurisprudencia/).

Arquivos:
* [doc.csv](juris_tcu/doc.csv) - cada documento contém um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/).
* [query1.csv](juris_tcu/query1.csv) - 50 queries geradas a partir do log de acessos à [Pesquisa Integrada do TCU](https://pesquisa.apps.tcu.gov.br/) (consultas mais executadas)
* [query2.csv](juris_tcu/query2.csv) - transformação das perguntas em query3 para expressão de busca, retirando parte das palavras utilzadas 
* [query3.csv](juris_tcu/query3.csv) - 50 queries, cada qual gerada por LLM a partir de um enunciado, dentre os mais acessados no log de acessos à [Pesquisa Integrada do TCU](https://pesquisa.apps.tcu.gov.br/)
* qrel.csv  ...![WIP](../docs/image/work-in-progress-thumbnail.png)

## [JURIS_TCU_INDEX](/data/juris_tcu_index/)
Base de indexação dos enunciados da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/) por termos do [Vocabulário de Controle Externo do Tribunal de Contas da União (VCE)](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).

Arquivos:
* [doc.csv](juris_tcu_index/doc.csv) - cada documento contém a definição de um termo do [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).
* [query.csv](data/juris_tcu_index/query.csv) - cada query é um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/).
* [qrel.csv](data/juris_tcu_index/qrel.csv) - cada registro corresponde a uma indexação de um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/) por um termo do [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm). Essa indexação foi realizada por operadores do sistema de Jurisprudência, e pode ser observada na [Pesquisa Integrada do TCU - base de Jurisprudência Selecionada](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada).

## Dados de acesso à pesquisa da Jurisprudência
Elaborados a partir do log de acessos à [Pesquisa Integrada do TCU](https://pesquisa.apps.tcu.gov.br/), entre junho/2022 a maio/2023.

Apenas consultas específicas à base de [Jurisprudência Selecionada](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada), sem o uso de [operadores de proximidade](https://portal.tcu.gov.br/data/files/F4/F4/F0/B2/223648102DFE0FF7F18818A8/Manual_Resumido_Pesquisa_Jurisprudencia_TCU.pdf).

Arquivos:
* [query.csv](log_juris_tcu/query.csv) - buscas efetuadas, em ordem decrescente de quantidade de execuções.
  * Contém a expressão de busca (query), a quantidade de execuções (count), e a média de documentos retornados (#docs).
* [doc-hits.csv](log_juris_tcu/doc-hits.csv) - documentos acessados, em ordem decrescente de quantidade de acessos.
  * Contém identificador (ID), chave de pesquisa (KEY), quantidade de acessos (COUNT) e posição média nos resultados das buscas (AVG_POSITION).
  * O documento pode ser encontrado na [Pesquisa Integrada do TCU](https://pesquisa.apps.tcu.gov.br/) pela chave, exemplo: [JURISPRUDENCIA-SELECIONADA-2845](https://pesquisa.apps.tcu.gov.br/resultado/jurisprudencia-selecionada/JURISPRUDENCIA-SELECIONADA-2845.KEY)
* [query-doc-hits.csv](log_juris_tcu/query-doc-hits.csv) - cruzamento dos documentos com as expressões de busca utilizadas.
  * Contém identificador (ID), chave de pesquisa (KEY), quantidade de acessos ao documento a partir da consulta (COUNT) e a expressão de busca (QUERY).
  
## Preparação do Dataset [JURIS_TCU](/data/juris_tcu/) por LLM
Arquivos intermediários para formação das queries para o dataset [JURIS_TCU](/data/juris_tcu/), gerados a partir de LLM.

Arquivos:
* [query_llm.txt](llm_juris_tcu/query_llm.txt) - perguntas produzidas pelo [ChatGPT](https://openai.com/chatgpt) a partir dos enunciados mais acessados da [Jurisprudência Selecionada](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada)
* [query_llm_selecionada.txt](llm_juris_tcu/query_llm_selecionada.txt) - curadoria manual realizada a partir do arquivo anterior, cada registro contém 2 versões de queries:
  * uma pergunta completa
  * uma expressão de pesquisa

## Resultados dos experimentos de busca realizados
* Indexação da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/) por termos do [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm): [JURIS_TCU_INDEX](/data/search/juris_tcu_index/).
* Buscas efetuadas sobre queries do DataSet [JURIS_TCU](/data/juris_tcu/): [JURIS_TCU](/data/search/juris_tcu/).

## Outros links
* [Página principal do projeto](/README.md)
* Esta página, [em inglês](./README.md)
