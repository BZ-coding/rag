from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    model_name="llama-3-chinese-8b-instruct-v3-f16",
    openai_api_key="your-api-key",
    openai_api_base="http://localhost:11434/v1/",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

messages = [
    (
        "system", "You are a helpful assistant.",
    ),
    ("user", "Say this is a test."),
]
ai_msg = llm.invoke(messages)

print(ai_msg)
print(ai_msg.content)
