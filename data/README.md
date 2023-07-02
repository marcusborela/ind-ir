# IndIR - Prepared Datasets and Other Files

Generated final datasets:
* [JURIS_TCU](#juris_tcu)
* [JURIS_TCU_INDEX](#juris_tcu_index)

Folders with intermediate results:
* [Jurisprudence Research Access Data](#jurisprudence-research-access-data)
* [LLM Preparation of the JURIS_TCU Dataset](#llm-preparation-of-the-juris_tcu-dataset)
* [Search Results for Statements Selection - JURIS_TCU_BASIC](#search-results-for-statements-selection---juris_tcu_basic)
* [Search Experiment Results - JURIS_TCU_INDEX](#search-experiment-results---juris_tcu_index)
* [Search Experiment Results - JURIS_TCU](#search-experiment-results---juris_tcu)

Other links:
* [Project Main Page](/README.md)
* This page, [in Portuguese](./leiame.md)

## [JURIS_TCU](/data/juris_tcu/)
Statement base of the [Selected Jurisprudence of the Federal Court of Accounts](https://portal.tcu.gov.br/jurisprudencia/), containing 16,057 documents, 150 queries, and 2,250 relevance assessments.

There are 15 document assessments per query, including:
* 10 [highest-ranked documents](llm_juris_tcu/eval_most_relevants.csv) from a complete search pipeline: top-300 BM-25 search, top-300 similarity search, document union, followed by reranking with monoT5;
* 5 [randomly selected documents](llm_juris_tcu/eval_least_relevants.csv) from BM-25 search, excluding the 10 documents from the previous group.

Relevance assessment is performed by [LLM ChatGPT 4.0](llm_juris_tcu/prompt_response.txt) on a scale of 0 to 3:
* 0 - irrelevant - the statement does not answer the question;
* 1 - related - the statement is only on the topic of the question;
* 2 - relevant - the statement partially answers the question;
* 3 - highly relevant - the statement fully answers the question, addressing all its nuances.

In addition to the score, LLM provides the reason for choosing the score. The complete evaluation report is available in [evaluations.txt](llm_juris_tcu/evaluations.txt).

Files:
* [doc.csv](juris_tcu/doc.csv) - each document contains a statement from the [Selected Jurisprudence](https://portal.tcu.gov.br/jurisprudencia/).
* [query.csv](juris_tcu/query.csv) - 150 generated queries, divided in 3 groups of 50 queries each:
  * Group 1 - search expressions generated from the access log of the [Integrated TCU Search](https://pesquisa.apps.tcu.gov.br/) (most executed queries).
  * Group 2 - transformation of the questions in group 3 into search expressions, removing part of the used words.
  * Group 3 - generated questions by LLM from a statement among the most accessed in the log of the [Integrated TCU Search](https://pesquisa.apps.tcu.gov.br/).
* [qrel.csv](juris_tcu/qrel.csv).csv - assessments of 15 documents for each  query from [query.csv](juris_tcu/query.csv).

## [JURIS_TCU_INDEX](/data/juris_tcu_index/)
Indexing base of statements from the [Selected Jurisprudence](https://portal.tcu.gov.br/jurisprudencia/) by terms from the [External Control Vocabulary of the Federal Court of Accounts (VCE)](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).

Files:
* [doc.csv](juris_tcu_index/doc.csv) - each document contains the definition of a term from the [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).
* [query.csv](data/juris_tcu_index/query.csv) - each query is a statement from the [Selected Jurisprudence](https://portal.tcu.gov.br/jurisprudencia/).
* [qrel.csv](data/juris_tcu_index/qrel.csv) - each record corresponds to an indexing of a statement from the [Selected Jurisprudence](https://portal.tcu.gov.br/jurisprudencia/) by a term from the [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm). This indexing was performed by operators of the Jurisprudence system and can be observed in the [Integrated TCU Search - Selected Jurisprudence database](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada).

## [Jurisprudence Research Access Data](/data/log_juris_tcu/)
Prepared from the access log of the [Integrated TCU Search](https://pesquisa.apps.tcu.gov.br/), between June 2022 and May 2023.

Only specific queries to the [Selected Jurisprudence](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada) database, without the use of [proximity operators](https://portal.tcu.gov.br/data/files/F4/F4/F0/B2/223648102DFE0FF7F18818A8/Manual_Resumido_Pesquisa_Jurisprudencia_TCU.pdf).

Files:
* [query.csv](log_juris_tcu/query.csv) - executed searches, in descending order of execution quantity.
  * Contains the search expression (query), the execution quantity (count), and the average number of returned documents (#docs).
* [doc-hits.csv](log_juris_tcu/doc-hits.csv) - accessed documents, in descending order of access quantity.
  * Contains the identifier (ID), search key (KEY), access quantity (COUNT), and average position in search results (AVG_POSITION).
  * The document can be found in the [Integrated TCU Search](https://pesquisa.apps.tcu.gov.br/) using the key, for example: [JURISPRUDENCIA-SELECIONADA-2845](https://pesquisa.apps.tcu.gov.br/resultado/jurisprudencia-selecionada/JURISPRUDENCIA-SELECIONADA-2845.KEY)
* [query-doc-hits.csv](log_juris_tcu/query-doc-hits.csv) - cross-referencing of documents with the used search expressions.
  * Contains the identifier (ID), search key (KEY), access quantity to the document from the query (COUNT), and the search expression (QUERY).
  
## Preparation of the [JURIS_TCU](/data/juris_tcu/) Dataset by LLM
Intermediate files for query formation for the [JURIS_TCU](/data/juris_tcu/) dataset, generated from a Large Language Model, specifically ChatGPT 4.0.

Files:
* [query_llm.txt](llm_juris_tcu/query_llm.txt) - questions produced by [ChatGPT](https://openai.com/chatgpt) based on the most accessed statements from the [Selected Jurisprudence](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada).
* [query_llm_selecionada.txt](llm_juris_tcu/query_llm_selecionada.txt) - manual curation performed based on the previous file, each record contains 2 versions of queries:
  * a complete question
  * a search expression
* [evaluations.txt](llm_juris_tcu/evaluations.txt) - relevance assessment report conducted by LLM, containing the score and reason for the choice, for each of the 150 queries from the database and 15 selected statements per query, totaling 2,250 evaluations.
* [eval_least_relevants.csv](llm_juris_tcu/eval_least_relevants) - selection of the top 10 highest-ranked documents for each of the 150 queries, using a complete search pipeline: top-300 BM-25 search, top-300 similarity search, document join, followed by reranking (monoT5). A total of 1,500 documents, including the relevance assessment by LLM, containing the score and reason for the choice.
* [eval_most_relevants.csv](llm_juris_tcu/eval_most_relevants.csv) - selection of 5 randomly chosen documents from BM-25 search, excluding the 10 documents from the previous group. A total of 750 documents, including the relevance assessment by LLM, containing the score and reason for the choice.
* [eval_statistics.csv](llm_juris_tcu/eval_statistics.csv) - evaluation statistics per query, containing the quantity of evaluations for each score.
* [eval_statistics_groups.csv](llm_juris_tcu/eval_statistics_groups.csv) - evaluation statistics for each group of 50 queries.
* [query-generation.csv](llm_juris_tcu/query-generation.csv) - list of base statements for queries in groups 2 and 3, created from the file [query_llm_selecionada.txt](llm_juris_tcu/query_llm_selecionada.txt) and the spreadsheet [Juris-TCU-query-generation.xlsx](/docs/explanation/Juris-TCU-query-generation.xlsx).
* [prompt_response.txt](llm_juris_tcu/prompt_response.txt) - example of a prompt sent to LLM and the received response.

## Search Results for Statements Selection - [JURIS_TCU_BASIC](/data/search/juris_tcu_basic/)
Searches performed on queries from the [JURIS_TCU](/data/juris_tcu/) dataset to construct the evaluations (QREL). An intermediate step in the production of evaluations for the mentioned dataset.

Search reports - for each of the 150 queries from the [JURIS_TCU](/data/juris_tcu/) dataset, they record the top 10 ranked statements for each search pipeline:
* [results_bm25.txt](search/juris_tcu/results_bm25.txt) - BM25 search;
* [results_bm25_reranker.txt](search/juris_tcu/results_bm25_reranker.txt) - BM25 search with reranking of the top 300 by the monoT5 model;
* [results_sts.txt](search/juris_tcu/results_sts.txt) - dense search using vector similarity generated by the monoT5 model;
* [results_sts_reranker.txt](search/juris_tcu/results_sts_reranker.txt) - similarity search with reranking of the top 300 by the monoT5 model;
* [results_join_bm25_sts_reranker.txt](search/juris_tcu/results_join_bm25_sts_reranker.txt) - pipeline formed by 2 retrievers (BM25 and STS), where the top 300 results are joined and reranked by the monoT5 model.

Execution result files:
* [run_bm25.csv](search/juris_tcu/results_bm25) - BM25 search, up to the top 1000 results;
* [run_bm25_reranker.csv](search/juris_tcu/results_bm25_reranker) - BM25 search, up to the top 300 results ordered by monoT5 reranker;
* [run_sts.csv](search/juris_tcu/run_sts) - vector similarity search, up to the top 1000 results;
* [run_sts_reranker.csv](search/juris_tcu/results_join_bm25_sts_reranker) - vector similarity search, up to the top 300 results ordered by monoT5 reranker;
* [run_join_bm25_sts_reranker.csv](search/juris_tcu/results_sts) - pipeline formed by 2 retrievers (BM25 and STS), where the top 300 results are joined, generating a maximum of 600 results ordered by monoT5 reranker.

## Search Experiment Results - [JURIS_TCU_INDEX](/data/search/juris_tcu_index/)
Indexing of the [Selected Jurisprudence](https://portal.tcu.gov.br/jurisprudencia/) by terms from the [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).

## Search Experiment Results - [JURIS_TCU](/data/search/juris_tcu/)
Search Results over [JURIS_TCU](/data/juris_tcu/) dataset, for final results evaluation. 
