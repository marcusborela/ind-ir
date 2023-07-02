# IndIR - Datasets e outros arquivos preparados

Datasets finais gerados:
* [JURIS_TCU](#juris_tcu)
* [JURIS_TCU_INDEX](#juris_tcu_index)

Pastas com resultados intermediários:
* [Dados de acesso à pesquisa da Jurisprudência](#dados-de-acesso-à-pesquisa-da-jurisprudência)
* [Preparação do Dataset JURIS_TCU por LLM](#prepara%C3%A7%C3%A3o-do-dataset-juris_tcu-por-llm)
* [Resultados da busca para selecionar enunciados - JURIS_TCU_BASIC](#resultados-da-busca-para-selecionar-enunciados---juris_tcu_basic)
* [Resultados dos experimentos de busca - JURIS_TCU_INDEX](#resultados-dos-experimentos-de-busca---juris_tcu_index)
* [Resultados dos experimentos de busca - JURIS_TCU](#resultados-dos-experimentos-de-busca---juris_tcu)

Outros links:
* [Página principal do projeto](/README.md)
* Esta página, [em inglês](./README.md)

## [JURIS_TCU](/data/juris_tcu/)
Base de enunciados da [Jurisprudência Selecionada do Tribunal de Contas da União](https://portal.tcu.gov.br/jurisprudencia/), contendo 16.057 documentos, 150 consultas e 2.250 avaliações de relevância.

São 15 avaliações de documentos por consulta, das quais:
* 10 são os [documentos melhor ranqueados](llm_juris_tcu/eval_most_relevants.csv) a partir de pipeline completa de busca: top-300 de busca BM-25, top-300 de busca por similaridade, join (união) de documentos, seguido por rerank (monoT5); 
* 5 são [documentos escolhidos randomicamente](llm_juris_tcu/eval_least_relevants.csv) de busca BM-25, excetuados os 10 documentos do grupo anterior.

A avaliação de relevância é realizada pelo [LLM ChatGPT 4.0](llm_juris_tcu/prompt_response.txt), em uma escala de 0 a 3:
* 0 - irrelevante - o enunciado não responde a pergunta;
* 1 - relacionado - o enunciado apenas está no tópico da pergunta;
* 2 - relevante - o enunciado responde parcialmente a pergunta;
* 3 - altamente relevante - o enunciado responde a pergunta, tratando completamente de suas nuances.

Além do score, o LLM fornece a razão pela qual escolhe o score. Relatório completo com avaliações está em [evaluations.txt](llm_juris_tcu/evaluations.txt).

Arquivos:
* [doc.csv](juris_tcu/doc.csv) - cada documento contém um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/).
* [query.csv](juris_tcu/query.csv) - 150 queries produzidas. São 3 grupos de 50 queries cada:
  * Grupo 1 - expressões de busca geradas a partir do log de acessos à [Pesquisa Integrada do TCU](https://pesquisa.apps.tcu.gov.br/) (consultas mais executadas).
  * Grupo 2 - transformação das perguntas do grupo 3 para expressão de busca, retirando parte das palavras utilizadas.
  * Grupo 3 - perguntas geradas por LLM a partir de um enunciado, dentre os mais acessados no log de acessos à [Pesquisa Integrada do TCU](https://pesquisa.apps.tcu.gov.br/).
* [qrel.csv](juris_tcu/qrel.csv).csv - avaliações de 15 documentos para cada query de [query.csv](juris_tcu/query.csv).

## [JURIS_TCU_INDEX](/data/juris_tcu_index/)
Base de indexação dos enunciados da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/) por termos do [Vocabulário de Controle Externo do Tribunal de Contas da União (VCE)](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).

Arquivos:
* [doc.csv](juris_tcu_index/doc.csv) - cada documento contém a definição de um termo do [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).
* [query.csv](data/juris_tcu_index/query.csv) - cada query é um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/).
* [qrel.csv](data/juris_tcu_index/qrel.csv) - cada registro corresponde a uma indexação de um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/) por um termo do [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm). Essa indexação foi realizada por operadores do sistema de Jurisprudência, e pode ser observada na [Pesquisa Integrada do TCU - base de Jurisprudência Selecionada](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada).

## [Dados de acesso à pesquisa da Jurisprudência](/data/log_juris_tcu/)
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
Arquivos intermediários para formação das queries para o dataset [JURIS_TCU](/data/juris_tcu/), gerados a partir de Large Language Model, mais especificamente ChatGPT 4.0.

Arquivos:
* [query_llm.txt](llm_juris_tcu/query_llm.txt) - perguntas produzidas pelo [ChatGPT](https://openai.com/chatgpt) a partir dos enunciados mais acessados da [Jurisprudência Selecionada](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada)
* [query_llm_selecionada.txt](llm_juris_tcu/query_llm_selecionada.txt) - curadoria manual realizada a partir do arquivo anterior, cada registro contém 2 versões de queries:
  * uma pergunta completa
  * uma expressão de pesquisa
* [evaluations.txt](llm_juris_tcu/evaluations.txt) - relatório das avaliações de relevância realizadas pelo LLM, contendo score e a razão da escolha, para cada uma das 150 consultas da base e 15 enunciados selecionados por consulta, totalizando 2.250 avaliações.
* [eval_least_relevants.csv](llm_juris_tcu/eval_least_relevants) - seleção dos 10 documentos melhor ranqueados para cada uma das 150 queries, a partir de pipeline completa de busca: top-300 de busca BM-25, top-300 de busca por similaridade, join (união) de documentos, seguido por rerank (monoT5). Totaliza 1.500 documentos, contendo ainda a avaliação de relevância por LLM, contendo score e razão da escolha.
* [eval_most_relevants.csv](llm_juris_tcu/eval_most_relevants.csv) - seleção de 5 documentos escolhidos randomicamente de busca BM-25, excetuados os 10 documentos do grupo anterior. Totaliza 750 documentos, contendo ainda a avaliação de relevância por LLM, contendo score e razão da escolha.
* [eval_statistics.csv](llm_juris_tcu/eval_statistics.csv) - estatística de avaliações por query, contendo a quantidade de avaliações para cada score.
* [eval_statistics_groups.csv](llm_juris_tcu/eval_statistics_groups.csv) - estatística de avaliações por cada grupo de 50 queries.
* [query-generation.csv](llm_juris_tcu/query-generation.csv) - relação dos enunciados base das queries dos grupos 2 e 3, formada a partir do arquivo [query_llm_selecionada.txt](llm_juris_tcu/query_llm_selecionada.txt) e planilha [Juris-TCU-query-generation.xlsx](/docs/explanation/Juris-TCU-query-generation.xlsx).
* [prompt_response.txt](llm_juris_tcu/prompt_response.txt) - exemplo de prompt enviado ao LLM e a resposta recebida.

## Resultados da busca para selecionar enunciados - [JURIS_TCU_BASIC](/data/search/juris_tcu_basic/)
Buscas efetuadas sobre queries do dataset [JURIS_TCU](/data/juris_tcu/), para construção das avaliações (QREL). Passo intermediário para a produção das avaliações para o referidado dataset.

Relatórios de busca - para cada uma das 150 queries do dataset [JURIS_TCU](/data/juris_tcu/), registram os 10 enunciados melhor rankeados por cada pipeline de busca:
* [results_bm25.txt](search/juris_tcu/results_bm25.txt) - busca BM25;
* [results_bm25_reranker.txt](search/juris_tcu/results_bm25_reranker.txt) - busca BM25 cujos top-300 são rerankeados por modelo monoT5;
* [results_sts.txt](search/juris_tcu/results_sts.txt) - busca densa, por similaridade de vetores gerados por modelo monoT5;
* [results_sts_reranker.txt](search/juris_tcu/results_sts_reranker.txt) - busca por similaridade cujos top-300 são rerankeados por modelo monoT5;
* [results_join_bm25_sts_reranker.txt](search/juris_tcu/results_join_bm25_sts_reranker.txt) - pipeline formado por 2 retrievers (BM25 e STS), cujos top-300 são juntados (união) e rerankeados por modelo monoT5.

Arquivos de resultado de execução: 
* [run_bm25.csv](search/juris_tcu/results_bm25) - busca BM25, até 1000 melhores resultados;
* [run_bm25_reranker.csv](search/juris_tcu/results_bm25_reranker) - busca BM25, até 300 melhores resultados ordenados por reranker monoT5;
* [run_sts.csv](search/juris_tcu/run_sts) - busca por similaridade de vetores, até 1000 melhores resultados;
* [run_sts_reranker.csv](search/juris_tcu/results_join_bm25_sts_reranker) - busca por similaridade de vetores, até 300 melhores resultados ordenados por reranker monoT5;
* [run_join_bm25_sts_reranker.csv](search/juris_tcu/results_sts) - pipeline formado por 2 retrievers (BM25 e STS), cujos top-300 são juntados (união), gerando no máximo 600 resultados ordenados por reranker monoT5.

## Resultados dos experimentos de busca - [JURIS_TCU_INDEX](/data/search/juris_tcu_index/)
Indexação da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/) por termos do [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).

## Resultados dos experimentos de busca - [JURIS_TCU](/data/search/juris_tcu/)
Resultados de busca sobre dataset [JURIS_TCU](/data/juris_tcu/), para avaliação final de resultados. 
