from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import logging
import telebot

from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import initialize_agent, AgentType
from langchain.tools.base import StructuredTool

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

llm = Ollama(
    model="command-r",
    base_url="http://10.2.4.87:11434",
    keep_alive=-1,
    system="Ты - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Отвечай на сообщения и вопросы пользователя максимально дружелюбно. Твоя основная цель - получить у пользователя его локацию и телефон для последующего создания заявки. Тебе доступен набор инструментов. Настоятельно рекомендуется их использовать для выполнения основной цели.",
    num_ctx=2048,
    repeat_last_n=-1,
    temperature=0.3,
)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         MessagesPlaceholder("chat_history"),
#         ("user", "{input}"),
#     ]
# )
# llm_chain = LLMChain(llm=llm, prompt=prompt)


def create_location_tool(bot, chat_id):
    def send_location_request():
        keyboard = telebot.types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
        button = telebot.types.KeyboardButton(
            text="Send your location", request_location=True
        )
        keyboard.add(button)
        bot.send_message(
            chat_id, "Please submit your location below", reply_markup=keyboard
        )

    return StructuredTool.from_function(
        func=send_location_request,
        name="Request Location",
        description="Используй разово, когда тебе нужно запросить локацию пользователя для дальнейшего использования этой информации при создании заявки, чтобы помочь пользователю.",
    )


def create_contact_tool(bot, chat_id):
    def send_contact_request():
        keyboard = telebot.types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
        button = telebot.types.KeyboardButton(
            text="Send your contact", request_contact=True
        )
        keyboard.add(button)
        bot.send_message(
            chat_id, "Please submit your contact below", reply_markup=keyboard
        )

    return StructuredTool.from_function(
        func=send_contact_request,
        name="Request Contact",
        description="Используй разово, когда тебе нужно запросить контакты пользователя для дальнейшего использования этой информации при создании заявки, чтобы помочь пользователю.",
    )


chat_history = {}

@app.post("/message")
async def call_message(request: Request, authorization: str = Header(None)):
    logger.info("post: message")
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    else:
        answer = "Failed to get bot token"
        return JSONResponse(content={"type": "text", "body": str(answer)})

    if token:
        message = await request.json()
        bot = telebot.TeleBot(token)

        chat_id = message["chat"]["id"]
        user_message = message["text"]

        if user_message == "/reset":
            chat_history[chat_id] = []
            bot.send_message(chat_id, "Chat history has been reset")

        if chat_id not in chat_history:
            chat_history[chat_id] = []

        if user_message != "/start" and user_message != "/reset":
            location_tool = create_location_tool(bot, chat_id)
            contact_tool = create_contact_tool(bot, chat_id)
            tools = [location_tool, contact_tool]

            agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
            )
            # keyboard = telebot.types.ReplyKeyboardMarkup(
            #     row_width=1, resize_keyboard=True
            # )
            # button = telebot.types.KeyboardButton(
            #     text="Send your location",
            #     request_location=True,
            # )
            # keyboard.add(button)

            # bot.send_message(
            #     chat_id, "Please submit your location below", reply_markup=keyboard
            # )

            try:
                # bot_response = llm_chain.invoke(
                #     {"chat_history": chat_history[chat_id], "input": user_message}
                # )["text"]
                bot_response = agent.run(
                    input=user_message, chat_history=chat_history[chat_id]
                )["output"]
            except Exception as e:
                logger.info(f"Error: {e}")

            chat_history[chat_id].extend(
                [HumanMessage(content=user_message), AIMessage(content=bot_response)]
            )
            logger.info(f"History for {chat_id}: {chat_history[chat_id]}")

            bot.send_message(chat_id, bot_response)
