from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import logging
import telebot
import os

# from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

from langchain.agents import initialize_agent, AgentType

from config_manager import ConfigManager
from file_service import FileService

from location_tool import create_location_tool
from contact_tool import create_contact_tool

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# llm = Ollama(
#     model="command-r",
#     base_url="http://10.2.4.87:11434",
#     keep_alive=-1,
#     system="Вы - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Отвечайте на сообщения и вопросы пользователя максимально дружелюбно. Ваша основная итоговая цель - получить у пользователя его локацию и телефон для последующего создания заявки и отвечать сообщениями на сообщения пользователя. Вам доступен набор инструментов. Используйте только один инструмент за раз, если он ещё не был использован. Отвечайте на РУССКОМ языке, учитывая контекст переписки.",
#     num_ctx=8192,
#     repeat_last_n=4096,
#     temperature=0.2,
# )

config_manager = ConfigManager("config.json")

chat_history_service = FileService(config_manager.get("chats_dir"), logger)
request_service = FileService(config_manager.get("request_dir"), logger)

llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    model=config_manager.get("model"),
    temperature=config_manager.get("temperature"),
)


@app.post("/message")
async def call_message(request: Request, authorization: str = Header(None)):
    logger.info("post: message")

    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    else:
        answer = "Не удалось получить токен бота"
        return JSONResponse(content={"type": "text", "body": str(answer)})

    if token:
        message = await request.json()
        bot = telebot.TeleBot(token)
        chat_id = message["chat"]["id"]

        logger.info(f"Message: {message}")

        if "location" in message:
            user_message = f'Мои координаты - {message["location"]}'
            await request_service.save_to_request(
                chat_id, message["location"], message["message_id"], "address"
            )
        elif "contact" in message:
            user_message = f'Мой телефон - {message["contact"]["phone_number"]}'
            await request_service.save_to_request(
                chat_id, message["contact"]["phone_number"], message["message_id"], "phone"
            )
        elif "text" in message:
            user_message = message["text"]
        else:
            return JSONResponse(content={"type": "empty", "body": ""})

        if user_message == "/reset":
            chat_history_service.delete_chat_history(chat_id)
            bot.send_message(chat_id, "История чата была очищена")

        if user_message != "/start" and user_message != "/reset":
            location_tool = create_location_tool(bot, chat_id)
            contact_tool = create_contact_tool(bot, chat_id)
            tools = [location_tool, contact_tool]

            agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
            )

            request = await request_service.read_request(chat_id)

            message_text = f"Вы - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Ваша первочередная итоговая цель - для создания заявки запросить у пользователя его локацию и телефон путем ИСПОЛЬЗОВАНИЯ предоставленных вам инструментов и отвечать сообщениями на сообщения пользователя. Вам доступен набор инструментов. Вам НАСТОЯТЕЛЬНО рекомендуется использовать ваши инструменты. Используйте ТОЛЬКО ОДИН инструмент за раз, если он ещё не был использован. Текущее содержание заявки: {request}. Отвечайте на РУССКОМ языке, учитывая контекст переписки. Сейчас вы получили следующее сообщение: {user_message}"

            chat_history = await chat_history_service.read_chat_history(chat_id)
            logger.info(f"History for {chat_id}: {chat_history}")

            try:
                bot_response = agent.run(input=message_text, chat_history=chat_history)

                await chat_history_service.save_to_chat_history(
                    chat_id,
                    user_message,
                    message["message_id"],
                    "HumanMessage",
                    message["from"]["first_name"],
                    "human",
                )
                await chat_history_service.save_to_chat_history(
                    chat_id,
                    bot_response,
                    message["message_id"],
                    "AIMessage",
                    message["from"]["first_name"],
                    "llm",
                )
                bot.send_message(chat_id, bot_response)

                chat_history = await chat_history_service.read_chat_history(chat_id)
                logger.info(f"History for {chat_id}: {chat_history}")

            except Exception as e:
                logger.info(f"Error: {e}")
