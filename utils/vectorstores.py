from langchain_community.vectorstores import FAISS


class FaissVectorStore:
    def __init__(self, text_embedding_pairs, embedding_model):
        self.faiss = FAISS.from_embeddings(text_embedding_pairs, embedding_model)
        self.retriever = self.faiss.as_retriever()

    def similarity_search(self, query):
        return self.faiss.similarity_search(query)
