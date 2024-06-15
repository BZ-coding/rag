import os
import pickle

from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def get_embedding_model(model_path, model_kwargs=None, encode_kwargs=None,
                        query_instruction="为这个句子生成表示以用于检索相关文章："):
    # https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding
    if model_kwargs is None:
        model_kwargs = {'device': 'cuda'}
    if encode_kwargs is None:
        encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction=query_instruction
    )
    if query_instruction:
        embedding_model.query_instruction = query_instruction

    return embedding_model


def get_text_embedding_pairs(embedding_model, data, text_embedding_pairs_path=None):
    if text_embedding_pairs_path and os.path.exists(text_embedding_pairs_path):
        return pickle.load(open(text_embedding_pairs_path, 'rb'))

    data_embeddings = embedding_model.embed_documents(data)
    text_embedding_pairs = zip(data, data_embeddings)
    if text_embedding_pairs_path:
        pickle.dump(text_embedding_pairs, open(text_embedding_pairs_path, 'wb'))
    return text_embedding_pairs
