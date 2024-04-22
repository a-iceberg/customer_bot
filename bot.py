import os
import logging

import telebot
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse

from config_manager import ConfigManager
from file_service import FileService
from langchain_env import ChatAgent

class Application:
    def __init__(self):
        self.config_manager = ConfigManager("./data/config.json")
        self.logger = self.setup_logging()
        self.set_keys()
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

    def set_keys(self):
        cm = ConfigManager("./data/keys.json")

        os.environ["LANGCHAIN_API_KEY"] = cm.get("LANGCHAIN_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = cm.get("OPENAI_API_KEY", "")
        os.environ["ANTHROPIC_API_KEY"] = cm.get("ANTHROPIC_API_KEY", "")
        os.environ["1С_TOKEN"] = cm.get("1С_TOKEN", "")
        self.logger.info("Keys set successfully")

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

                message_text = f'Вы - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Ваша первочередная итоговая цель - для создания заявки запросить сообщениями у пользователя по одному адрес неисправности и контактный телефон, сохранить их с помощью ваших инструментов и в принципе вежливо вести диалог с пользователем, когда это приемлемо. Вам доступен набор инструментов. Вам рекомендуется использовать ваши инструменты, когда вы сочтете нужным. Текущее содержание заявки: {request}. Если в заявке не хватает адреса или телефона, запрашивайте их по отдельности. Адрес нужен либо в строгом формате Город, улица, номер дома, либо прикрепленный GPS-координатами в диалоге, донесите это в том числе до пользователя. После получения от пользователя сообщения с этими данными ИСПОЛЬЗУЙТЕ ОДИН из ваших инструментов для сохранения данных в заявку, в зависимости от того, что именно было получено. Если в заявке уже есть и "phone", и "address", ОБЯЗАТЕЛЬНО СНАЧАЛА уточните у пользователя корректность СРАЗУ ВСЕХ переданных им данных, содержащихся в заявке, прислав их ему, по отдельности НЕ уточняйте. И ТОЛЬКО ПОТОМ, в случае получения ПОДТВЕРЖДЕНИЯ, ИСПОЛЬЗУЙТЕ ваш инструмент "Создание заявки". Подтверждением может быть и простое "да". Если же пользователь указал на неточность данных, снова запрашивайте актуальные данные и вызывайте инструменты для обновления заявки. Отвечайте на русском языке, учитывая контекст переписки. На благодарности пользователя просто свободно отвечайте, а не ещё раз уточняйте данные. Не здоровайтесь повторно посреди диалога. Сейчас вы получили от пользователя следующее сообщение: {user_message}'

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
