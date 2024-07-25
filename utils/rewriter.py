from time import sleep

from .chatbot import ChatBot

# DEFAULT_SYSTEM_PROMPT = """你是一个非常有效的query改写员，你要先对query进行意图识别，然后针对最有可能的意图，进行query改写。请注意query可能的省略及笔误等情况。
# 你的所改写的query将会被传输给rag文档检索或ai agent工具链。
# 请将你的回答思考过程一步一步的回答出来，并在最后另起一行，只返回你认为最有效的一个改写后的query。
# 请你将回答保持与原query一样的语言。"""
DEFAULT_SYSTEM_PROMPT = """你是一个非常有效的query改写员，你要先对query进行意图识别，然后针对最有可能的意图，进行query改写。请注意query可能的省略及笔误等情况。
你的所改写的query将会被传输给rag文档检索或ai agent工具链。
请不要将你的思考过程回答出来，只返回你认为最有效的一个改写后的query。
请你将回答保持与原query一样的语言。"""
DEFAULT_USER_PROMPT = """请你对“{query}”这个query进行改写。"""


class ReWriter:
    def __init__(self, chatbot, system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT):
        self.chatbot = chatbot
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def rewrite(self, query, stream=False):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(query=query)},
        ]
        response = self.chatbot.chat(messages=messages, stream=stream)
        return response


if __name__ == "__main__":
    chatbot = ChatBot()
    rewriter = ReWriter(chatbot=chatbot)

    for token in rewriter.rewrite(query="贩毒一百克有什么后果？", stream=True):
        print(token, end='', flush=True)
        sleep(0.1)
    print('\n')
