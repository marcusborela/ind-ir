# Projeto indIR - Datasets e outros arquivos preparados durante o projeto

## [JURIS_TCU](/data/juris_tcu/)
Base de enunciados da [Jurisprudência Selecionada do Tribunal de Contas da União](https://portal.tcu.gov.br/jurisprudencia/).
* [doc.csv](juris_tcu/doc.csv) - cada documento contém um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/).
* query.csv - ![WIP](../docs/image/work-in-progress-thumbnail.png)
* qrel.csv - ![WIP](../docs/image/work-in-progress-thumbnail.png)

## [JURIS_TCU_INDEX](/data/juris_tcu_index/)
Base de indexação dos enunciados da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/) por termos do [Vocabulário de Controle Externo do Tribunal de Contas da União (VCE)](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm).
* [doc.csv](juris_tcu_index/doc.csv) - cada documento contém a definição de um termo do [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm)
* [query.csv](data/juris_tcu_index/query.csv) - cada query é um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/).
* [qrel.csv](data/juris_tcu_index/qrel.csv) - cada registro corresponde a uma indexação de um enunciado da [Jurisprudência Selecionada](https://portal.tcu.gov.br/jurisprudencia/) por um termo do [VCE](https://portal.tcu.gov.br/vocabulario-de-controle-externo-do-tribunal-de-contas-da-uniao-vce.htm). Essa indexação foi realizada por operadores do sistema de Jurisprudência, e pode ser observada na [Pesquisa Integrada do TCU - base de Jurisprudência Selecionada](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada)

## Outros arquivos produzidos
* [log_juris_tcu](/data/log_juris_tcu/) - arquivos elaborados a partir do log de acessos à [Pesquisa Integrada do TCU](https://pesquisa.apps.tcu.gov.br/), base de [Jurisprudência Selecionada](https://pesquisa.apps.tcu.gov.br/pesquisa/jurisprudencia-selecionada)
* [search](/data/search/) - resultados dos experimentos de busca realizados

## Outros links
[Página principal do projeto](/README.md)
