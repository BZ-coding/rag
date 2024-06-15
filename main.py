"""
docker run --rm -i -t --runtime=nvidia --gpus all nvcr.io/nvidia/pytorch:24.04-py3 bash
docker exec -it gpu_env bash
"""

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain_core.runnables import RunnableLambda

from utils.embeddings import get_embedding_model, get_text_embedding_pairs
from utils.vectorstores import FaissVectorStore
from utils.rerankers import Reranker
from utils.rag import Rag

EMBEDDING_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/embedding_models/BAAI/bge-large-zh-v1.5/"
TEXT_EMBEDDING_PAIRS_PATH = 'text_embedding_pairs_BAAI_1.pkl'
RERANKER_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/reranker_models/BAAI/bge-reranker-large/"
CHAT_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/chinese-alpaca-2-7b/"

if __name__ == '__main__':
    with open("刑法.txt", "r") as f:
        data = f.readlines()
    data = [d.strip() for d in data]
    # print(f"data sample : {data[:2]}")
    # print(f"data length : {len(data)}")

    embedding_model = get_embedding_model(model_path=EMBEDDING_MODEL_PATH)
    # print(f"embedding model : {embedding_model}")

    text_embedding_pairs = get_text_embedding_pairs(embedding_model=embedding_model, data=data,
                                                    text_embedding_pairs_path=TEXT_EMBEDDING_PAIRS_PATH)

    vectorstore = FaissVectorStore(text_embedding_pairs=text_embedding_pairs, embedding_model=embedding_model,
                                   search_kwargs={"k": 3})

    chat_model = LlamaForCausalLM.from_pretrained(
        CHAT_MODEL_PATH,
        # load_in_8bit=True,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    tokenizer = LlamaTokenizer.from_pretrained(CHAT_MODEL_PATH)

    rag = Rag(chat_model=chat_model, tokenizer=tokenizer, retriever=vectorstore.retriever)

    query = "持有管制刀具怎么判？"
    print(rag.answer(query=query))

    print("\n\n\n")
    text_embedding_pairs = get_text_embedding_pairs(embedding_model=embedding_model, data=data,
                                                    text_embedding_pairs_path=TEXT_EMBEDDING_PAIRS_PATH)
    vectorstore = FaissVectorStore(text_embedding_pairs=text_embedding_pairs, embedding_model=embedding_model,
                                   search_kwargs={"k": 20})
    reranker = Reranker(ranker_model_path=RERANKER_MODEL_PATH, retriever=vectorstore.retriever, topn=3)
    rag = Rag(chat_model=chat_model, tokenizer=tokenizer, retriever=RunnableLambda(reranker.invoke))
    print(rag.answer(query=query))
