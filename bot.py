from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import logging
import telebot

from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

llm = Ollama(
    model="command-r",
    base_url="http://10.2.4.87:11434",
    keep_alive=-1,
    template="Ты - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Отвечай на сообщения и вопросы пользователя максимально дружелюбно",
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ты - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Отвечай на сообщения и вопросы пользователя максимально дружелюбно. Ответь на следующее сообщение."),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ]
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

chat_history = []

@app.post("/message")
async def call_message(request: Request, authorization: str = Header(None)):
    logger.info("post: message")
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    else:
        answer = "Не удалось определить токен бота."
        return JSONResponse(content={"type": "text", "body": str(answer)})

    if token:
        message = await request.json()
        logger.info(f"message: {message}")
        bot = telebot.TeleBot(token)
        chat_id = message["chat"]["id"]
        user_message = message["text"]

        try:
            bot_response = llm_chain.invoke({"chat_history": chat_history, "input": user_message})
        except Exception as e:
            logger.info(f"Error: {e}")

        logger.info(f"response: {bot_response}")

        chat_history.append(HumanMessage(content=user_message))
        chat_history.append(AIMessage(content=bot_response))

        logger.info(f"history: {chat_history}")

        bot.send_message(chat_id, bot_response)
