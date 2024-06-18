from langchain_community.vectorstores import FAISS


class FaissVectorStore:
    def __init__(self, text_embedding_pairs, embedding_model):
        self.faiss = FAISS.from_embeddings(text_embedding_pairs, embedding_model)

    def similarity_search(self, query):
        return self.faiss.similarity_search(query)

    def get_retriever(self, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {}
        return self.faiss.as_retriever(search_kwargs=search_kwargs)
