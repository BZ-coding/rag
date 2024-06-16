from langchain.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import pipeline


class Rag:
    def __init__(self, chat_model, tokenizer, retriever):
        template = """你是问答任务助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。
        Question: {question} 
        Context: {context} 
        Answer:
        """
        pipe = pipeline(
            "text-generation",
            model=chat_model,
            tokenizer=tokenizer,
            max_length=4096,
            truncation=True,
            repetition_penalty=1.2,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )
        self.local_llm = HuggingFacePipeline(pipeline=pipe)
        prompt = ChatPromptTemplate.from_template(template)
        self.rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.local_llm
                | StrOutputParser()
        )

    def answer(self, query):
        return self.rag_chain.invoke(query)
