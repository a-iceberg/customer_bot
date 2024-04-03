from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import logging
import telebot

from langchain_community.llms import Ollama
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
    create_stuff_documents_chain,
)
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
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Основываясь на предоставленном диалоге, создай поисковый запрос, чтобы получить информацию, релевантную диалогу",
        ),
    ]
)
retriever_chain = create_history_aware_retriever(llm, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ответь на вопрос пользователя основываясь на предоставленном контексте:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

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

        chat_history.append(HumanMessage(content=user_message))

        bot_response = retrieval_chain.invoke(
            {"chat_history": chat_history, "input": user_message}
        )

        chat_history.append(AIMessage(content=bot_response))

        bot.send_message(chat_id, bot_response)
