from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # required but ignored
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='llama-3-chinese-8b-instruct-v3-f16',
)

print(chat_completion)
print(chat_completion.choices[0].message.content)
