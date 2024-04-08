from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import logging
import telebot

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import initialize_agent, AgentType
from langchain.tools.base import StructuredTool

from config_manager import ConfigManager
from chat_history_service import ChatHistoryService

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

llm = Ollama(
    model="command-r",
    base_url="http://10.2.4.87:11434",
    keep_alive=-1,
    system="Ты - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Отвечай на сообщения и вопросы пользователя максимально дружелюбно. Твоя основная итоговая цель - получить у пользователя его локацию и телефон для последующего создания заявки и отвечать сообщениями на сообщения пользователя. Тебе доступен набор инструментов. Настоятельно рекомендуется использовать их все для выполнения основной цели. Отвечай на РУССКОМ языке, учитывая контекст переписки.",
    num_ctx=8192,
    repeat_last_n=4096,
    temperature=0.3,
)

config_manager = ConfigManager("config.json")
chat_history_service = ChatHistoryService(config_manager.get("chats_dir"), logger)

def create_location_tool(bot, chat_id):
    def send_location_request():
        keyboard = telebot.types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
        button = telebot.types.KeyboardButton(
            text="Ваше местоположение", request_location=True
        )
        keyboard.add(button)
        bot.send_message(
            chat_id,
            "Укажите ваше местоположение через кнопку ниже",
            reply_markup=keyboard,
        )
        return "Локация пользователя была успешно запрошена"

    return StructuredTool.from_function(
        func=send_location_request,
        name="Request Location",
        description="Используй, когда тебе нужно запросить локацию пользователя для дальнейшего использования этой информации при создании заявки, чтобы помочь пользователю.",
        return_direct=False,
    )


def create_contact_tool(bot, chat_id):
    def send_contact_request():
        keyboard = telebot.types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
        button = telebot.types.KeyboardButton(text="Ваш телефон", request_contact=True)
        keyboard.add(button)
        bot.send_message(
            chat_id, "Укажите ваш телефон через кнопку ниже", reply_markup=keyboard
        )
        return "Контакты пользователя были успешно запрошены"

    return StructuredTool.from_function(
        func=send_contact_request,
        name="Request Contact",
        description="Используй, когда тебе нужно запросить контакты пользователя для дальнейшего использования этой информации при создании заявки, чтобы помочь пользователю.",
        return_direct=False,
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

        if "text" not in message:
            return JSONResponse(content={"type": "empty", "body": ""})

        user_message = message["text"]

        if user_message != "/start":
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

            message_text = f"Ты - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Отвечай на сообщения и вопросы пользователя максимально дружелюбно. Твоя основная итоговая цель - получить у пользователя его локацию и телефон для последующего создания заявки и отвечать сообщениями на сообщения пользователя. Тебе доступен набор инструментов. Настоятельно рекомендуется использовать их все для выполнения основной цели. Отвечай на РУССКОМ языке, учитывая контекст переписки. Сейчас ты получил следующее сообщение: {user_message}"

            try:
                chat_history = await chat_history_service.read_chat_history(chat_id)
                logger.info(f"History for {chat_id}: {chat_history}")
            except:
                await chat_history_service.save_to_chat_history(
                    chat_id,
                    "",
                    message["message_id"],
                    "InitialMessage",
                    message["from"]["first_name"],
                )
                chat_history = await chat_history_service.read_chat_history(chat_id)

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
                chat_history = await chat_history_service.read_chat_history(chat_id)
                logger.info(f"History for {chat_id}: {chat_history}")
                bot.send_message(chat_id, bot_response)

            except Exception as e:
                logger.info(f"Error: {e}")
