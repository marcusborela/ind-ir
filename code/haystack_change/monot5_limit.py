"""
Code created by Marcus Vinícius Borela de Castro in the context of IND-IR project
https://github.com/marcusborela/ind-ir
"""
from typing import List, Optional, Union, Any, Iterable, Mapping, Tuple
import logging
from pathlib import Path
from copy import deepcopy
import torch
from torch.nn import DataParallel
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizer, PreTrainedModel
from dataclasses import dataclass
from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker
from haystack.modeling.utils import initialize_device_settings

logger = logging.getLogger(__name__)

TokenizerReturnType = Mapping[str, Union[torch.Tensor, List[int],
                                         List[List[int]],
                                         List[List[str]]]]

DecodedOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

prediction_tokens = {
        'castorini/monot5-base-msmarco':          ['▁false', '▁true'],
        'castorini/monot5-base-msmarco-10k':      ['▁false', '▁true'],
        'castorini/monot5-large-msmarco':         ['▁false', '▁true'],
        'castorini/monot5-large-msmarco-10k':     ['▁false', '▁true'],
        'castorini/monot5-base-med-msmarco':      ['▁false', '▁true'],
        'castorini/monot5-3b-med-msmarco':        ['▁false', '▁true'],
        'castorini/monot5-3b-msmarco-10k':           ['▁false', '▁true'],
        'unicamp-dl/mt5-base-en-msmarco':            ['▁no'   , '▁yes'],
        'unicamp-dl/ptt5-base-msmarco-pt-10k':    ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-msmarco-pt-100k':   ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-msmarco-en-pt-10k': ['▁não'  , '▁sim'],
        'unicamp-dl/mt5-base-multi-msmarco':      ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-en-pt-msmarco-v1':      ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-en-pt-msmarco-v2':   ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-mmarco-v1':          ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-mmarco-v2':          ['▁no'   , '▁yes'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v1': ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-10k-v1':    ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-10k-v2':    ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2':   ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não'  , '▁sim'],
        'unicamp-dl/mt5-3B-mmarco-en-pt':   ['▁'  , '▁true'],
        'unicamp-dl/mt5-13b-mmarco-100k':            ['▁', '▁true'],
        # a confirmar
        'unicamp-dl/ptt5-base-pt-msmarco-10k-v2': ['▁no'  , '▁yes'],
        # a confirmar
        'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2': ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-1400': ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-2200': ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-7600': ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-lim100-1700': ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-lim50-800': ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-lim50-2200': ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-41-pcte':  ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-73-pcte':  ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-79-pcte':  ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-83-pcte':  ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2-indir-106-pcte':  ['▁não'  , '▁sim'],
        }


class Query:
    """Class representing a query.
    A query contains the query text itself and potentially other metadata.

    Parameters
    ----------
    text : str
        The query text.
    id : Optional[str]
        The query id.
    """

    def __init__(self, text: str, id: Optional[str] = None): # pylint: disable=redefined-builtin, invalid-name # código de Query externa
        self.text = text
        self.id = id # pylint: disable= invalid-name # código de Query externa



@dataclass
class QueryDocumentBatch: # pylint: disable=missing-class-docstring # código de origem externa
    query: Query
    documents: List[Document]
    output: Optional[TokenizerReturnType] = None

    def __len__(self):
        return len(self.documents)

class Text:
    """Class representing a text to be reranked.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.

    Parameters
    ----------
    text : str
        The text to be reranked.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the text. For example, the score might be the BM25 score
        from an initial retrieval stage.
    title : Optional[str]
        The text's title.
    """

    def __init__(self,
                 text: str,
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 title: Optional[str] = None):
        self.text = text
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.title = title

class TokenizerEncodeMixin: # pylint: disable=missing-class-docstring # código de origem externa
    tokenizer: PreTrainedTokenizer = None
    tokenizer_kwargs = None

    def encode(self, strings: List[str]) -> TokenizerReturnType: # pylint: disable=missing-function-docstring # código de origem externa
        assert self.tokenizer and self.tokenizer_kwargs is not None, \
                'mixin used improperly'
        ret = self.tokenizer.batch_encode_plus(strings,
                                               **self.tokenizer_kwargs)
        ret['tokens'] = list(map(self.tokenizer.tokenize, strings))
        return ret


class QueryDocumentBatchTokenizer(TokenizerEncodeMixin): # pylint: disable=missing-class-docstring # código de origem externa
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,
                 pattern: str = '{query} {document}',
                 **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer_kwargs = tokenizer_kwargs
        # alterado para ajuste da relevância
        self.pattern = "Query: {query} Document: {document} Relevant:"
        # self.pattern = pattern

    def traverse_query_document( # pylint: disable=missing-function-docstring # código de origem externa
            self,
            batch_input: QueryDocumentBatch) -> Iterable[QueryDocumentBatch]:
        query = batch_input.query
        for batch_idx in range(0, len(batch_input), self.batch_size):
            docs = batch_input.documents[batch_idx:batch_idx + self.batch_size]
            outputs = self.encode([self.pattern.format(
                                        query=query.text,
                                        document=doc.content) for doc in docs])
            yield QueryDocumentBatch(query, docs, outputs)

class T5BatchTokenizer(QueryDocumentBatchTokenizer): # pylint: disable=missing-class-docstring # código de origem externa
    def __init__(self, *args, **kwargs):
        kwargs['pattern'] = 'Query: {query} Document: {document} Relevant:'
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512
        super().__init__(*args, **kwargs)


@torch.no_grad()
def greedy_decode(model: PreTrainedModel, # pylint: disable=missing-function-docstring # código de origem externa
                  input_ids: torch.Tensor,
                  length: int,
                  attention_mask: torch.Tensor = None,
                  return_last_logits: bool = True) -> DecodedOutput:
    decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True)
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat([decode_ids,
                                next_token_logits.max(1)[1].unsqueeze(-1)],
                               dim=-1)
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids


def get_prediction_tokens(pretrained_model_name_or_path: str, # pylint: disable=missing-function-docstring
        tokenizer, token_false, token_true):
    # para retirar modelos/ que vem antes do nome
    ultimo_indice = pretrained_model_name_or_path.rfind('/')
    penultimo_indice = pretrained_model_name_or_path.rfind('/', 0, ultimo_indice)
    if penultimo_indice > 0:
        model_name = pretrained_model_name_or_path[penultimo_indice + 1:]
    else:
        model_name =  pretrained_model_name_or_path
    if not (token_false and token_true):
        if model_name in prediction_tokens:
            token_false, token_true = prediction_tokens[model_name]
            token_false_id = tokenizer.tokenizer.get_vocab()[token_false]
            token_true_id  = tokenizer.tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id
        else:
            raise Exception("We don't know the indexes for the non-relevant/relevant tokens for\
                    the checkpoint {model_name} and you did not provide any.")
    else:
        token_false_id = tokenizer.tokenizer.get_vocab()[token_false]
        token_true_id  = tokenizer.tokenizer.get_vocab()[token_true]
        return token_false_id, token_true_id


class MonoT5RankerLimit(BaseRanker):
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
        top_k: Optional[int] = 0,
        use_gpu: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        batch_size: int = 16,
        use_amp = False, # Automatic Mixed Precision
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        limit_query_size:int=350,
        token_false = None,
        token_true  = None,
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
            self.top_k = 0
        else:
            self.top_k = top_k

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)

        self.progress_bar = progress_bar
        self.transformer_model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        ).to(str(self.devices[0])).eval()
        self.tokenizer = T5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path, use_fast=False,
                                          revision=model_version, use_auth_token=use_auth_token), batch_size=batch_size)

        if len(self.devices) > 1:
            self.model = DataParallel(self.transformer_model, device_ids=self.devices)

        self.batch_size = batch_size
        self.max_position_embeddings =  512 # self.transformer_model.config.max_position_embeddings - 2 # separadores entre sentenças?
        assert limit_query_size is not None, f"limit_query_size must be set, not None!"
        self.limit_query_size = limit_query_size
        self.use_amp = use_amp
        self.token_false_id, self.token_true_id = get_prediction_tokens(pretrained_model_name_or_path=model_name_or_path, tokenizer=self.tokenizer, token_false=token_false, token_true=token_true)


    def return_num_token(self, parm_texto:str):
        return len(self.tokenizer.tokenizer.tokenize(parm_texto))


    def return_text_limited_num_token(self, parm_texto: str, parm_num_limite_token: int):
        tokens = self.tokenizer.tokenizer.tokenize(parm_texto)
        if len(tokens) > parm_num_limite_token:
            tokens = tokens[:parm_num_limite_token]
        text_limited = self.tokenizer.tokenizer.convert_tokens_to_string(tokens)
        return text_limited


    def return_text_limited_num_token_ultima_pontuacao(self, parm_texto: str, parm_num_limite_token: int):
        tokens = self.tokenizer.tokenizer.tokenize(parm_texto)

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
                tokens = tokens[:pos_ultima_pontuacao]
            text_limited = self.tokenizer.tokenizer.convert_tokens_to_string(tokens)
            return text_limited


    def rescore(self, query: Query, texts: List[Document]) -> List[Document]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.devices[0])
                attn_mask = batch.output['attention_mask'].to(self.devices[0])
                _, batch_scores = greedy_decode(self.transformer_model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()

            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score


        return texts

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

        # updated 2023-06-23
        # before: text= query
        query_obj = Query(text= query_limited)

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


        """Sorts a list of texts
        """
        if top_k == 0:
            return sorted(self.rescore(query_obj, documents_limited_size), key=lambda x: x.score, reverse=True)
        else:
            return sorted(self.rescore(query_obj, documents_limited_size), key=lambda x: x.score, reverse=True)[:top_k]

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        raise Exception('It is not coded!')
