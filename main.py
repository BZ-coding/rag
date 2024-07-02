"""
docker run --rm -i -t --runtime=nvidia --gpus all nvcr.io/nvidia/pytorch:24.04-py3 bash
docker exec -it gpu_env bash
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.embeddings import get_embedding_model, get_text_embedding_pairs
from utils.vectorstores import FaissVectorStore
from utils.rerankers import Reranker
# from utils.rag import Rag
from utils.rag import MyRag as Rag

EMBEDDING_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/embedding_models/BAAI/bge-large-zh-v1.5/"
TEXT_EMBEDDING_PAIRS_PATH = 'text_embedding_pairs_BAAI.pkl'
RERANKER_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/reranker_models/BAAI/bge-reranker-large/"
# CHAT_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/chinese-alpaca-2-7b/"
CHAT_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/llama-3-chinese-8b-instruct-v3/"


def answer(vectorstore, chat_model, tokenizer, query, ranker_model_path=None):
    if ranker_model_path:
        retriever = vectorstore.get_retriever(search_kwargs={"k": 20})
        reranker = Reranker(ranker_model_path=RERANKER_MODEL_PATH, retriever=retriever, topn=3)
        retriever = reranker.as_retriever()
    else:
        retriever = vectorstore.get_retriever(search_kwargs={"k": 3})
    rag = Rag(chat_model=chat_model, tokenizer=tokenizer, retriever=retriever)
    return rag.answer(query=query)


if __name__ == '__main__':
    with open("刑法.txt", "r") as f:
        data = f.readlines()
    data = [d.strip() for d in data]

    embedding_model = get_embedding_model(model_path=EMBEDDING_MODEL_PATH)
    text_embedding_pairs = get_text_embedding_pairs(embedding_model=embedding_model, data=data,
                                                    text_embedding_pairs_path=TEXT_EMBEDDING_PAIRS_PATH)
    vectorstore = FaissVectorStore(text_embedding_pairs=text_embedding_pairs, embedding_model=embedding_model)

    chat_model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL_PATH,
        # load_in_8bit=True,
        device_map='auto',
        torch_dtype=torch.float16,  # 推理时用fp16精度更高，训练时要用bf16不容易精度溢出
    )
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_PATH)

    query = "持有管制刀具怎么判？"
    r=answer(vectorstore=vectorstore, chat_model=chat_model, tokenizer=tokenizer, query=query)
    print(r)

    print("\n\n\n")
    answer(vectorstore=vectorstore, chat_model=chat_model, tokenizer=tokenizer, query=query,
           ranker_model_path=RERANKER_MODEL_PATH)
