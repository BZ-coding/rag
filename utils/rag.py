from langchain.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain
from transformers import pipeline, TextStreamer


class Rag:
    def __init__(self, chat_model, tokenizer, retriever, streaming=True):
        streamer = None
        if streaming:
            streamer = TextStreamer(tokenizer)
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
            streamer=streamer,
        )
        self.local_llm = HuggingFacePipeline(pipeline=pipe)
        message = [
            {"role": "system",
             "content": "You are a helpful assistant. 你是一个乐于助人的助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。"},
            {"role": "user", "content": "Question: {question}\n\nContext: {context}"},
        ]
        template = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        prompt = ChatPromptTemplate.from_template(template)
        self.rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.local_llm
                | StrOutputParser()
        )

    def answer(self, query):
        return self.rag_chain.invoke(query)


class MyRag:
    def __init__(self, chat_model, tokenizer, retriever, streaming=True):
        streamer = None
        if streaming:
            streamer = TextStreamer(tokenizer)
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
            streamer=streamer,
            return_full_text=False,
        )
        self.local_llm = HuggingFacePipeline(pipeline=pipe)

        message = [
            {"role": "system",
             "content": "你是一个分析师。请从本文中提取与'{question}'相关的关键事实，不相关的可以舍弃。不要包括意见。给每个事实一个数字，并保持简短的句子。"},
            {"role": "user", "content": "Context: {context}"},
        ]
        template = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        fact_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        fact_extraction_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | fact_template
                | self.local_llm
                | StrOutputParser()
        )

        message = [
            {"role": "system", "content": "You are a helpful assistant. 你是一个乐于助人的助手。"},
            {"role": "user",
             "content": "Facts: {facts}\n\n请根据以上事实清单，写一个简短的段落来回答问题：{question}。注意不要脱离事实清单。"},
        ]
        template = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        answer_template = PromptTemplate(
            input_variables=["facts", "question"],
            template=template,
        )

        self.rag_chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                    "facts": fact_extraction_chain,
                }
                | answer_template
                | self.local_llm
                | StrOutputParser()
        )

    def answer(self, query):
        return self.rag_chain.invoke(query)
