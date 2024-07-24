from time import sleep

from openai import OpenAI, NOT_GIVEN

OPENAI_BASE_URL = 'http://localhost:11434/v1/'
MODEL_NAME = 'llama-3-chinese-8b-instruct-v3-f16'


# MODEL_NAME = 'qwen2:7b'

class ChatBot:
    def __init__(self):
        self.client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key='ollama',  # required but ignored
        )

    def _run_conversation(self, messages: list, temperature, stream, stop):
        response = self.client.chat.completions.create(
            messages=messages,
            temperature=temperature,
            stream=stream,
            model=MODEL_NAME,
            stop=stop,
        )
        return response

    def _chat(self, messages: list, temperature, stop):
        response = self._run_conversation(messages=messages, temperature=temperature, stream=False, stop=stop)
        return response.choices[0].message.content

    def _stream_chat(self, messages: list, temperature, stop):
        response = self._run_conversation(messages=messages, temperature=temperature, stream=True, stop=stop)
        for token in response:
            if token.choices[0].finish_reason is not None:
                continue
            yield token.choices[0].delta.content

    def chat(self, messages: list, temperature=0.6, stop=NOT_GIVEN, stream=False):
        if not stream:
            return self._chat(messages=messages, temperature=temperature, stop=stop)
        else:
            return self._stream_chat(messages=messages, temperature=temperature, stop=stop)


if __name__ == '__main__':
    chatbot = ChatBot()
    message = [{"role": "user", "content": "hello."}]
    print(chatbot.chat(messages=message))

    print("\n\n\n")

    for token in chatbot.chat(messages=message, stream=True):
        print(token, end='', flush=True)
        sleep(0.1)
    print('\n')
