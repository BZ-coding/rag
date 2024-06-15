from langchain_community.vectorstores import FAISS


class FaissVectorStore:
    def __init__(self, text_embedding_pairs, embedding_model, search_kwargs=None):
        self.faiss = FAISS.from_embeddings(text_embedding_pairs, embedding_model)

        if search_kwargs is None:
            search_kwargs = {}
        self.retriever = self.faiss.as_retriever(search_kwargs=search_kwargs)

    def similarity_search(self, query):
        return self.faiss.similarity_search(query)
