"""
Code created by Marcus Vinícius Borela de Castro in the context of IND-IR project
https://github.com/marcusborela/ind-ir
"""
from typing import List, Optional, Union, Tuple, Iterator, Any
import logging
from pathlib import Path

import torch
from torch.nn import DataParallel
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker
from haystack.modeling.utils import initialize_device_settings

logger = logging.getLogger(__name__)



class SentenceTransformersRankerLimit(BaseRanker):
    """
    Sentence Transformer based pre-trained Cross-Encoder model for Document Re-ranking (https://huggingface.co/cross-encoder).
    Re-Ranking can be used on top of a retriever to boost the performance for document search. This is particularly useful if the retriever has a high recall but is bad in sorting the documents by relevance.

    SentenceTransformerRanker handles Cross-Encoder models
        - use a single logit as similarity score e.g.  cross-encoder/ms-marco-MiniLM-L-12-v2
        - use two output logits (no_answer, has_answer) e.g. deepset/gbert-base-germandpr-reranking
    https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#usage-with-transformers

    With a SentenceTransformersRanker, you can:
     - directly get predictions via predict()

    Usage example:

    ```python
    retriever = BM25Retriever(document_store=document_store)
    ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")
    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=ranker, name="Ranker", inputs=["ESRetriever"])
    ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        model_version: Optional[str] = None,
        top_k: Optional[int] = 0, # todos são retornados
        use_gpu: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        batch_size: int = 16,
        scale_score: bool = True,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        limit_query_size:int=350
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g.
        'cross-encoder/ms-marco-MiniLM-L-12-v2'.
        See https://huggingface.co/cross-encoder for full list of available models
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param top_k: The maximum number of documents to return
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of documents to process at a time.
        :param scale_score: The raw predictions will be transformed using a Sigmoid activation function in case the model
                            only predicts a single label. For multi-label predictions, no scaling is applied. Set this
                            to False if you do not want any scaling of the raw predictions.
        :param progress_bar: Whether to show a progress bar while processing the documents.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__()

        if top_k is None:
            self.top_K = 0
        else:
            self.top_k = top_k

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)

        self.progress_bar = progress_bar
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        self.transformer_model.to(str(self.devices[0]))
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        self.transformer_model.eval()
        assert limit_query_size is not None, f"limit_query_size must be set, not None!"
        self.limit_query_size = limit_query_size

        # we use sigmoid activation function to scale the score in case there is only a single label
        # we do not apply any scaling when scale_score is set to False
        num_labels = self.transformer_model.num_labels
        self.activation_function: torch.nn.Module
        if num_labels == 1 and scale_score:
            self.activation_function = torch.nn.Sigmoid()
        else:
            self.activation_function = torch.nn.Identity()

        if len(self.devices) > 1:
            self.model = DataParallel(self.transformer_model, device_ids=self.devices)

        self.batch_size = batch_size
        self.max_position_embeddings =  self.transformer_model.config.max_position_embeddings - 2 # separadores entre sentenças?



    def return_num_token(self, parm_texto:str):
        return len(self.transformer_tokenizer.tokenize(parm_texto))


    def return_text_limited_num_token(self, parm_texto: str, parm_num_limite_token: int):
        tokens = self.transformer_tokenizer.tokenize(parm_texto)
        if len(tokens) > parm_num_limite_token:
            tokens = tokens[:parm_num_limite_token]
        text_limited = self.transformer_tokenizer.convert_tokens_to_string(tokens)
        return text_limited


    def return_text_limited_num_token_ultima_pontuacao(self, parm_texto: str, parm_num_limite_token: int):
        tokens = self.transformer_tokenizer.tokenize(parm_texto)

        if len(tokens) <= parm_num_limite_token:
            return parm_texto
        else:
            tokens = tokens[:parm_num_limite_token]

            # Encontra a última pontuação antes do limite de tokens
            ultimo_token_pontuacao = None
            pos_ultima_pontuacao = len(tokens)
            for i, token in enumerate(reversed(tokens)):
                if token in "!),.:;>?]}" :
                    ultimo_token_pontuacao = token
                    pos_ultima_pontuacao -= (i + 1)
                    break
            if ultimo_token_pontuacao is not None:
                tokens = tokens[:pos_ultima_pontuacao + 1]
            text_limited = self.transformer_tokenizer.convert_tokens_to_string(tokens)
            return text_limited


    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = 0) -> List[Document]:
        """
        Use loaded ranker model to re-rank the supplied list of Document.

        Returns list of Document sorted by (desc.) similarity with the query.

        :param query: Query string
        :param documents: List of Document to be re-ranked
        :param top_k: The maximum number of documents to return
        :return: List of Document
        """
        # self.limit_query_size = 350
        if top_k is None:
            top_k = self.top_k

        num_tokens_query = self.return_num_token(query)
        if num_tokens_query > self.limit_query_size:
            query_limited = self.return_text_limited_num_token_ultima_pontuacao(query, self.limit_query_size)
            # print(f"Query passed limit in {num_tokens_query - self.limit_query_size} tokens")
            # print(f"Before:  {query}")
            # print(f"Now:  {query_limited}")
            num_tokens_query = self.limit_query_size
        else:
            query_limited = query

        lista_num_tokens_docto = [self.return_num_token(doc.content) for doc in documents]
        # num_doc = len(lista_num_tokens_docto)
        # print(f"num_tokens_query {num_tokens_query}")
        # print(f"{num_doc} documentos em {lista_num_tokens_docto}")

        documents_limited_size = documents.copy()
        for pos, doc in enumerate(documents_limited_size):
            num_excesso = (lista_num_tokens_docto[pos] + num_tokens_query) - self.max_position_embeddings
            if num_excesso > 0:
                # doc_antes = doc.content
                doc.content = self.return_text_limited_num_token_ultima_pontuacao(doc.content, self.max_position_embeddings-num_tokens_query)
                # print(f"Doc {doc.id}  passed limit in {num_excesso} tokens")
                # print(f"Before:  {doc_antes}")
                # print(f"Now:  {doc.content}")

        # updated 2023-06-23
        # before: [query for doc in documents],

        features = self.transformer_tokenizer(
            [query_limited for doc in documents],
            [doc.content for doc in documents_limited_size],
            padding=True,
            truncation=True,
            max_length= self.max_position_embeddings,
            return_tensors="pt",
        ).to(self.devices[0])

        # SentenceTransformerRanker uses:
        # 1. the logit as similarity score/answerable classification
        # 2. the logits as answerable classification  (no_answer / has_answer)
        # https://www.sbert.net/docs/pretrained-models/ce-msmarco.html#usage-with-transformers
        with torch.inference_mode():
            similarity_scores = self.transformer_model(**features).logits

        logits_dim = similarity_scores.shape[1]  # [batch_size, logits_dim]
        sorted_scores_and_documents = sorted(
            zip(similarity_scores, documents),
            key=lambda similarity_document_tuple:
            # assume the last element in logits represents the `has_answer` label
            similarity_document_tuple[0][-1] if logits_dim >= 2 else similarity_document_tuple[0],
            reverse=True,
        )

        # add normalized scores to documents
        if top_k == 0:
            sorted_documents = self._add_scores_to_documents(sorted_scores_and_documents, logits_dim)
        else:
            sorted_documents = self._add_scores_to_documents(sorted_scores_and_documents[:top_k], logits_dim)
        return sorted_documents

    def _add_scores_to_documents(
        self, sorted_scores_and_documents: List[Tuple[Any, Document]], logits_dim: int
    ) -> List[Document]:
        """
        Normalize and add scores to retrieved result documents.

        :param sorted_scores_and_documents: List of score, Document Tuples.
        :param logits_dim: Dimensionality of the returned scores.
        """
        sorted_documents = []
        for raw_score, doc in sorted_scores_and_documents:
            if logits_dim >= 2:
                score = self.activation_function(raw_score)[-1]
            else:
                score = self.activation_function(raw_score)[0]

            doc.score = score.detach().cpu().numpy().tolist()
            sorted_documents.append(doc)

        return sorted_documents

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        raise Exception('It is not coded!')
