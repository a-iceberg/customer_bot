from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
import logging
import telebot
# import os

# # from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain.agents import initialize_agent, AgentType

from config_manager import ConfigManager
from file_service import FileService

from langchain_env import ChatAgent

# app = FastAPI()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()

# config_manager = ConfigManager("config.json")

# chat_history_service = FileService(config_manager.get("chats_dir"), logger)
# request_service = FileService(config_manager.get("request_dir"), logger)

# # llm = ChatOpenAI(
# #     openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
# #     model=config_manager.get("model"),
# #     temperature=config_manager.get("temperature"),
# # )

# llm = ChatAnthropic(
#     anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
#     model=config_manager.get("model"),
#     temperature=config_manager.get("temperature"),
# )


# @app.post("/message")
# async def call_message(request: Request, authorization: str = Header(None)):
#     logger.info("post: message")

#     token = None
#     if authorization and authorization.startswith("Bearer "):
#         token = authorization.split(" ")[1]

#     else:
#         answer = "Не удалось получить токен бота"
#         return JSONResponse(content={"type": "text", "body": str(answer)})

#     if token:
#         message = await request.json()
#         bot = telebot.TeleBot(token)
#         chat_id = message["chat"]["id"]

#         logger.info(f"Message: {message}")

#         if "location" in message:
#             geolocator = Nominatim(user_agent="my_app")
#             location = message["location"]
#             address = geolocator.reverse(
#                 f'{location["latitude"]}, {location["longitude"]}'
#             ).address
#             user_message = f"Мой адрес - {address}"

#         elif "text" in message:
#             user_message = message["text"]
#         else:
#             return JSONResponse(content={"type": "empty", "body": ""})

#         if user_message == "/reset":
#             chat_history_service.delete_chat_history(chat_id)
#             bot.send_message(chat_id, "История чата была очищена")

#         if user_message != "/start" and user_message != "/reset":
#             request = await request_service.read_request(chat_id)

#             location_tool = create_location_tool(chat_id, message)
#             contact_tool = create_contact_tool(chat_id, message)
#             request_tool = create_request_tool(request)
#             tools = [location_tool, contact_tool, request_tool]

#             agent = initialize_agent(
#                 tools,
#                 llm,
#                 agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#                 verbose=True,
#                 handle_parsing_errors=True,
#             )

#             message_text = f'Вы - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Ваша первочередная итоговая цель - для создания заявки запросить сообщениями у пользователя адрес неисправности и контактный телефон, сохранить их с помощью ваших инструментов и в принципе отвечать сообщениями на сообщения пользователя. Вам доступен набор инструментов. Вам НАСТОЯТЕЛЬНО рекомендуется использовать ваши инструменты, когда вы сочтете нужным. Текущее содержание заявки: {request}. Если в заявке не хватает адреса или телефона, запрашивайте их. Адрес нужен в формате Город, улица, номер дома, донесите это в том числе до пользователя. После получения от пользователя сообщения с этими данными ИСПОЛЬЗУЙТЕ один из ваших инструментов для сохранения данных в заявку. Не передавайте в эти инструменты какой-либо input. Если в заявке уже есть и "phone", и "address", ОБЯЗАТЕЛЬНО уточните у пользователя корректность переданных им данных, содержащихся в заявке, прислав их ему. И ТОЛЬКО ТОГДА, в случае получения подтверждения, ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙТЕ ваш инструмент "Создание заявки". Если же пользователь указал на неточность данных, снова запрашивайте актуальные данные для обновления заявки. Отвечайте на РУССКОМ языке, учитывая контекст переписки. Сейчас вы получили следующее сообщение: {user_message}'

#             chat_history = await chat_history_service.read_chat_history(chat_id)
#             logger.info(f"History for {chat_id}: {chat_history}")

#             try:
#                 bot_response = agent.run(input=message_text, chat_history=chat_history)

#                 await chat_history_service.save_to_chat_history(
#                     chat_id,
#                     user_message,
#                     message["message_id"],
#                     "HumanMessage",
#                     message["from"]["first_name"],
#                     "human",
#                 )
#                 await chat_history_service.save_to_chat_history(
#                     chat_id,
#                     bot_response,
#                     message["message_id"],
#                     "AIMessage",
#                     message["from"]["first_name"],
#                     "llm",
#                 )
#                 bot.send_message(chat_id, bot_response)

#                 chat_history = await chat_history_service.read_chat_history(chat_id)
#                 logger.info(f"History for {chat_id}: {chat_history}")

#             except Exception as e:
#                 logger.info(f"Error: {e}")


class Application:
    def __init__(self):
        self.config_manager = ConfigManager("config.json")
        self.logger = self.setup_logging()
        self.chat_history_service = FileService(
            self.config_manager.get("chats_dir"), self.logger
        )
        self.request_service = FileService(
            self.config_manager.get("request_dir"), self.logger
        )
        self.empty_response = JSONResponse(content={"type": "empty", "body": ""})

        self.app = FastAPI()
        self.setup_routes()

        self.chat_agent = None

    def text_response(self, text):
        return JSONResponse(content={"type": "text", "body": str(text)})

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def setup_routes(self):
        @self.app.post("/message")
        async def handle_message(request: Request, authorization: str = Header(None)):
            self.logger.info("handle_message")
            message = await request.json()
            self.chat_id = message["chat"]["id"]
            self.logger.info(message)

            token = None
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ")[1]

            if token:
                bot = telebot.TeleBot(token)
            else:
                answer = "Не удалось определить токен бота."
                return self.text_response(answer)

            if "location" in message:
                user_message = f"Мои координаты - {message['location']}"

            elif "text" in message:
                user_message = message["text"]
            else:
                return self.empty_response

            if user_message == "/reset":
                self.chat_history_service.delete_chat_history(self.chat_id)
                bot.send_message(self.chat_id, "История чата была очищена")

            elif user_message != "/start" and user_message != "/reset":
                request = await self.request_service.read_request(self.chat_id)

                message_text = f'Вы - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Ваша первочередная итоговая цель - для создания заявки запросить сообщениями у пользователя адрес неисправности и контактный телефон, сохранить их с помощью ваших инструментов и в принципе вести диалог с пользователем. Вам доступен набор инструментов. Вам настоятельно рекомендуется использовать ваши инструменты, когда вы сочтете нужным. Текущее содержание заявки: {request}. Если в заявке не хватает адреса или телефона, запрашивайте их по одному. Адрес нужен в формате Город, улица, номер дома, донесите это в том числе до пользователя. ТОЛЬКО ПОСЛЕ получения от ПОЛЬЗОВАТЕЛЯ сообщения с этими данными ИСПОЛЬЗУЙТЕ один из ваших инструментов для сохранения данных в заявку, в зависимости от того, что именно было получено. Если в заявке уже есть и "phone", и "address", ОБЯЗАТЕЛЬНО СНАЧАЛА уточните у пользователя корректность переданных им данных, содержащихся в заявке, ПРИСЛАВ их ему. И ТОЛЬКО ПОТОМ, в случае получения ПОДТВЕРЖДЕНИЯ, ИСПОЛЬЗУЙТЕ ваш инструмент "Создание заявки". Если же пользователь указал на неточность данных, снова запрашивайте актуальные данные и вызывайте инструменты для обновления заявки. Отвечайте на русском языке, учитывая контекст переписки. Сейчас вы получили от пользователя следующее сообщение: {user_message}'

                # Read chat history in LLM fromat
                chat_history = await self.chat_history_service.read_chat_history(
                    self.chat_id
                )
                self.logger.info(f"History for {self.chat_id}: {chat_history}")

                if self.chat_agent is None:
                    self.chat_agent = ChatAgent(
                        model=self.config_manager.get("model"),
                        temperature=self.config_manager.get("temperature"),
                        request_dir=self.config_manager.get("request_dir"),
                        base_url=self.config_manager.get("base_url"),
                        logger=self.logger,
                        bot_instance=bot,
                        chat_id=self.chat_id,
                    )
                    self.chat_agent.initialize_agent()

                bot_response = await self.chat_agent.agent.arun(
                    input=message_text, chat_history=chat_history
                )
                await self.chat_history_service.save_to_chat_history(
                    self.chat_id,
                    user_message,
                    message["message_id"],
                    "HumanMessage",
                    message["from"]["first_name"],
                    "human",
                )
                await self.chat_history_service.save_to_chat_history(
                    self.chat_id,
                    bot_response,
                    message["message_id"],
                    "AIMessage",
                    message["from"]["first_name"],
                    "llm",
                )

                self.logger.info("Replying in " + str(self.chat_id))
                self.logger.info(f"Answer: {bot_response}")
                return bot.send_message(self.chat_id, bot_response)


application = Application()
app = application.app
