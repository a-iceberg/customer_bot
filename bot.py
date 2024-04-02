from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import logging
import telebot

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@app.post("/message")
async def call_message(request: Request, authorization: str = Header(None)):
    logger.info("post: message")
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    if token:
        message = await request.json()
        logger.info(f"message: {message}")

        bot = telebot.TeleBot(token)

        chat_id = message["chat"]["id"]
        answer = message["text"]

        bot.send_message(chat_id, answer)

    else:
        answer = "Не удалось определить токен бота."
        return JSONResponse(content={"type": "text", "body": str(answer)})
