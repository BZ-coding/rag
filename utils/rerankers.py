# https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/README.md
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document


class Reranker:
    def __init__(self, ranker_model_path, retriever=None, topn=None):
        super().__init__()
        self.ranker = FlagReranker(ranker_model_path, use_fp16=True)
        self.retriever = retriever
        self.topn = topn

    def compute_score(self, query, answers):
        query_answer_pairs = [(query, answer) for answer in answers]
        return self.ranker.compute_score(query_answer_pairs)

    def rank(self, query, answers, topn=None):
        if len(answers) <= 1:
            return answers
        answers_str = answers
        if isinstance(answers[0], Document):
            answers_str = [answer.page_content for answer in answers]
        scores = self.compute_score(query, answers_str)
        sorted_answers = [answer for _, answer in sorted(zip(scores, answers), reverse=True)]
        if topn is None:
            topn = self.topn
        if topn:
            sorted_answers = sorted_answers[:topn]
        return sorted_answers

    def invoke(self, input: str, config=None, **kwargs):
        if self.retriever is None:
            raise RuntimeError('retriever is not initialized')
        sim_docs = self.retriever.invoke(input, config, **kwargs)
        sorted_answers = self.rank(query=input, answers=sim_docs)
        return sorted_answers
