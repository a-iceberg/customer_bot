import os
import re
import time
import json
import logging
import requests
import asyncio
import aiofiles
from uuid import uuid4
from pathlib import Path
from datetime import datetime

import telebot.async_telebot

from openai import OpenAI, RateLimitError
from pyrogram import Client
from pydub import AudioSegment
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, Header

import telebot
from telebot.types import ReplyKeyboardMarkup

from langchain_env import ChatAgent
from file_service import FileService
from config_manager import ConfigManager

class Application:
    def __init__(self):
        self.ban_manager = ConfigManager("./data/banned_users.json")
        self.config_manager = ConfigManager("./data/config.json")
        self.coordinates_manager = ConfigManager(
            "./data/affilates_coordinates.json"
        )
        self.logger = self.setup_logging()
        self.set_keys()
        self.chat_data_service = FileService(
            self.config_manager.get("chats_dir"),
            self.logger
        )
        self.request_service = FileService(
            self.config_manager.get("request_dir"),
            self.logger
        )
        self.empty_response = JSONResponse(
            content={"type": "empty", "body": ""}
        )
        self.app = FastAPI()
        self.setup_routes()
        self.chat_agent = None
        self.chat_history_client = None
        self.TOKEN = os.environ.get("BOT_TOKEN", "")
        self.CHANNEL_ID = os.environ.get("HISTORY_CHANNEL_ID", "")
        self.GROUP_ID = os.environ.get("HISTORY_GROUP_ID", "")
        self.channel_posts = {}
        self.banned_accounts = self.ban_manager.load_config()
        self.base_error_answer = "Извините, произошла ошибка в работе системы. Cвяжитесь с нами по телефону 8 495 723 723 0 для дальнейшей помощи."
        self.llm_error_answer = "Извините, произошла ошибка в работе системы. Попробуйте сформулировать ваше сообщение по-другому или же свяжитесь с нами по телефону 8 495 723 723 0 для дальнейшей помощи."

    def text_response(self, text):
        return JSONResponse(content={"type": "text", "body": str(text)})

    def set_keys(self):
        cm = ConfigManager("./data/auth.json")

        os.environ["LANGCHAIN_API_KEY"] = cm.get("LANGCHAIN_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = cm.get("OPENAI_API_KEY", "")
        os.environ["ANTHROPIC_API_KEY"] = cm.get("ANTHROPIC_API_KEY", "")
        os.environ["1С_TOKEN"] = cm.get("1С_TOKEN", "")
        os.environ["1C_LOGIN"] = cm.get("1C_LOGIN", "")
        os.environ["1C_PASSWORD"] = cm.get("1C_PASSWORD", "")
        os.environ["TELEGRAM_API_ID"] = cm.get("TELEGRAM_API_ID", "")
        os.environ["TELEGRAM_API_HASH"] = cm.get("TELEGRAM_API_HASH", "")
        os.environ["BOT_TOKEN"] = cm.get("BOT_TOKEN", "")
        os.environ["CHAT_HISTORY_TOKEN"] = cm.get("CHAT_HISTORY_TOKEN", "")
        os.environ["HISTORY_CHANNEL_ID"] = cm.get("HISTORY_CHANNEL_ID", "")
        os.environ["HISTORY_GROUP_ID"] = cm.get("HISTORY_GROUP_ID", "")
        os.environ["DB_USER"] = cm.get("DB_USER", "")
        os.environ["DB_PASSWORD"] = cm.get("DB_PASSWORD", "")
        os.environ["DB_HOST"] = cm.get("DB_HOST", "")
        os.environ["DB_PORT"] = cm.get("DB_PORT", "")
        os.environ["YANDEX_GEOCODER_KEY"] = cm.get("YANDEX_GEOCODER_KEY", "")

        self.logger.info("Auth data set successfully")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def setup_routes(self):
        @self.app.get("/test")
        def test():
            return self.text_response("ok") 

        @self.app.post("/message")
        async def handle_message(
            request: Request,
            authorization: str = Header(None)
        ):
            self.logger.info("handle_message")
            try:
                message = await request.json()
            except Exception as e:
                self.logger.error(f"Error in getting message: {e}")
                return await bot.send_message(
                    message["chat"]["id"],
                    self.base_error_answer
                )
            self.logger.info(message)

            if authorization and authorization.startswith("Bearer "):
                self.TOKEN = authorization.split(" ")[1]

            if self.TOKEN:
                server_api_uri = 'http://localhost:8081/bot{0}/{1}'
                telebot.apihelper.API_URL = server_api_uri
                self.logger.info(f'Setting API_URL: {server_api_uri}')

                server_file_url = 'http://localhost:8081'
                telebot.apihelper.FILE_URL = server_file_url
                self.logger.info(f'Setting FILE_URL: {server_file_url}')
                bot = telebot.async_telebot.AsyncTeleBot(self.TOKEN)
            else:
                self.logger.error("Failed to get bot token")
                return self.text_response("Не удалось определить токен")

            if message["chat"]["id"] == int(self.GROUP_ID) and "text" in message and "reply_to_message" in message:
                if message["text"] == "/ban":
                    banned_id = re.search(
                        r'Chat ID: (\d+)',
                        message["reply_to_message"]["text"]
                    ).group(1)
                    self.ban_manager.set(
                        banned_id,
                        time.strftime("%Y-%m-%d %H:%M", time.localtime())
                    )
                    self.banned_accounts = self.ban_manager.load_config()
                    try:
                        answer = await bot.send_message(
                            self.GROUP_ID,
                            f"Пользователь с chat_id {banned_id} был забанен",
                            reply_to_message_id=self.channel_posts[self.chat_id]
                        )
                        await asyncio.sleep(5)
                        await bot.delete_message(
                            self.GROUP_ID,
                            answer.message_id
                        )
                    except:
                        self.logger.info("Chat id not received yet")
                    self.logger.info(f"Banned user with chat_id {banned_id}")

                if message["text"] == "/unban":
                    try:
                        unbanned_id = re.search(
                            r'Chat ID: (\d+)',
                            message["reply_to_message"]["text"]
                        ).group(1)
                        self.ban_manager.delete(unbanned_id)
                        self.banned_accounts = self.ban_manager.load_config()
                        try:
                            answer = await bot.send_message(
                                self.GROUP_ID,
                                f"Пользователь с chat_id {unbanned_id} был разбанен",
                                reply_to_message_id=self.channel_posts[self.chat_id]
                            )
                            await asyncio.sleep(5)
                            await bot.delete_message(
                                self.GROUP_ID,
                                answer.message_id
                            )
                        except:
                            self.logger.info("Chat id not received yet")
                        self.logger.info(
                            f'Unbanned user with chat_id {unbanned_id}'
                        )
                    except:
                        answer = await bot.send_message(
                            self.GROUP_ID,
                            f"Пользователь с chat_id {unbanned_id} не был забанен",
                            reply_to_message_id=self.channel_posts[self.chat_id]
                        )
                        await asyncio.sleep(5)
                        await bot.delete_message(self.GROUP_ID, answer.message_id)
                        self.logger.info(
                            f"User with chat_id {unbanned_id} isn't banned"
                        )

            if message["from"]["first_name"] == "Telegram":
                if self.chat_id not in self.channel_posts:
                    if 'message_thread_id' in message:
                        self.channel_posts[self.chat_id] = message['message_thread_id']
                    else:
                        self.channel_posts[self.chat_id] = message["message_id"]

            if message["from"]["is_bot"] or message["from"]["first_name"] == "Telegram":
                return self.empty_response

            self.chat_id = message["chat"]["id"]
            self.message_id = message["message_id"]
            await self.chat_data_service.save_message_id(
                self.chat_id,
                self.message_id
            )

            if self.chat_id not in self.channel_posts:
                name = f'@{message["from"]["username"]}' if "username" in message["from"] else message["from"]["first_name"]
                await bot.send_message(
                    self.CHANNEL_ID,
                    f'Chat with {name} (Chat ID: {self.chat_id})'
                )

            if "location" in message:
                self.user_message = f"Мои координаты - {message['location']}"

            elif "text" in message:
                self.user_message = message["text"]

            elif (
                "audio" in message
                or "voice" in message
                or (
                    "document" in message
                    and "mime_type" in message["document"]
                    and "audio" in message["document"]["mime_type"]
                )
            ):
                if "audio" in message:
                    key = "audio"
                elif "voice" in message:
                    key = "voice"
                elif (
                    "document" in message
                    and "mime_type" in message["document"]
                    and "audio" in message["document"]["mime_type"]
                ):
                    key = "document"
                file_id = message[key]["file_id"]
                self.logger.info(f"Audiofile id: {file_id}")

                try:
                    file_info = await bot.get_file(file_id)
                    file_bytes = await bot.download_file(file_info.file_path)
                except Exception as e:
                    self.logger.error(f"Error downloading file: {e}")
                    return self.text_response("Пожалуйста, попробуйте текстом")

                self.logger.info(f"File_bytes: {len(file_bytes)}")
                if "audio" in message:
                    file_name = message[key]["file_name"]
                elif "voice" in message:
                    file_name = "temp.ogg"

                file_name = f"{uuid4().hex}_{file_name}"
                audio_path = os.path.join(
                    self.config_manager.get("audio_dir"),
                    str(self.chat_id)
                )
                Path(audio_path).mkdir(parents=True, exist_ok=True)
                file_path = os.path.join(audio_path, file_name)
                with open(file_path, "wb") as f:
                    f.write(file_bytes)

                try:
                    original_audio = AudioSegment.from_file(file_path)
                except Exception as e:
                    self.logger.error(f"Error loading audio file: {e}")
                    os.remove(file_path)
                    return self.text_response("Пожалуйста, попробуйте текстом")

                if " " in file_path:
                    self.logger.info(
                        f"Replacing space in audiofile path: {file_path}"
                    )
                    new_file_path = file_path.replace(" ", "_")
                    os.rename(file_path, new_file_path)
                    file_path = new_file_path
                    self.logger.info(f"New audiofile path: {file_path}")

                self.logger.info(f"Converting audio to {file_path}")
                converted_audio = (
                    original_audio.set_frame_rate(16000)
                    .set_channels(1)
                    .export(file_path, format="mp3")
                )

                self.logger.info("Transcribing audio..")
                try:
                    self.user_message = transcribe_audio_file(file_path)
                except Exception as e:
                    self.logger.error(f"Error transcribing audio file: {e}")
                    return self.text_response("Пожалуйста, попробуйте текстом")                
                self.logger.info("Transcription finished")
            else:
                return self.empty_response

            if self.user_message == "/start":
                await bot.delete_message(self.chat_id, self.message_id)
                self.request_service.delete_files(self.chat_id)
                await self.chat_data_service.update_chat_history_date(self.chat_id)

                markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
                markup.add("📝 Хочу оформить новую заявку")
                markup.add("📑 Выбрать свою активную заявку")
                welcome_message = (
                    "Здраствуйте, это сервисный центр. Чем могу вам помочь?"
                )
                await bot.send_message(
                    self.chat_id,
                    welcome_message,
                    reply_markup=markup
                )
            
            elif self.user_message == "📑 Выбрать свою активную заявку":
                await bot.delete_message(self.chat_id, self.message_id)
                token = os.environ.get("1С_TOKEN", "")
                login = os.environ.get("1C_LOGIN", "")
                password = os.environ.get("1C_PASSWORD", "")

                try:
                    ws_url = f"{self.config_manager.get('proxy_url')}/ws"        
                    ws_params = {
                        "Идентификатор": "bid_numbers",
                        "НомерПартнера": str(self.chat_id),
                    }
                    ws_data = {
                        "clientPath": self.config_manager.get("ws_paths"),
                        "login": login,
                        "password": password,
                    }
                    request_numbers = {}
                    divisions = self.config_manager.get("divisions")
                except Exception as e:
                    self.logger.error(f"Error in getting web service params: {e}")
                    return f"Ошибка при получении параметров вэб-сервиса: {e}"

                try:
                    results = requests.post(
                        ws_url,
                        json={
                            "config": ws_data,
                            "params": ws_params,
                            "token": token
                        }
                    ).json()["result"]
                    self.logger.info(f"results: {results}")
                    for value in results.values():
                        if len(value) > 0:
                            for request in value:
                                request_numbers[request["id"]] = {
                                    "date": request["date"][:10],
                                    "division": divisions[request["division"]]
                                }
                    self.logger.info(f"request_numbers: {request_numbers}")
                except Exception as e:
                    self.logger.error(
                        f"Error in receiving request numbers: {e}"
                    )

                if len(request_numbers) > 0:
                    markup = ReplyKeyboardMarkup(
                        resize_keyboard=True,
                        one_time_keyboard=True
                    )
                    for number, values in request_numbers.items():
                        markup.add(
                            f"Заявка {number} от {values['date']}; {values['division']}"
                        )
                    markup.add("🏠 Вернуться в меню")
                    await bot.send_message(
                        self.chat_id,
                        "Выберете нужную заявку ниже 👇",
                        reply_markup=markup
                    )
                else:
                    await bot.send_message(
                        self.chat_id,
                        "К сожалению, у вас нет текущих активных заявок. Буду рад помочь оформить новую! 😃"
                    )
            
            elif self.user_message =="🏠 Вернуться в меню":
                await bot.delete_message(self.chat_id, self.message_id)
                markup = ReplyKeyboardMarkup(
                    resize_keyboard=True,
                    one_time_keyboard=True
                )
                markup.add("📝 Хочу оформить новую заявку")
                markup.add("📑 Выбрать свою активную заявку")
                return_message = (
                    "Возвращаюсь в меню..."
                )
                await bot.send_message(
                    self.chat_id,
                    return_message,
                    reply_markup=markup
                )

            elif self.user_message == "/requestreset":
                await bot.delete_message(self.chat_id, self.message_id)
                self.request_service.delete_files(self.chat_id)
                answer = await bot.send_message(
                    self.chat_id,
                    "Информация по заявкам была очищена"
                )
                await asyncio.sleep(5)
                await bot.delete_message(self.chat_id, answer.message_id)

            elif self.user_message == "/fullreset":
                await bot.delete_message(self.chat_id, self.message_id)
                self.request_service.delete_files(self.chat_id)
                await self.chat_data_service.update_chat_history_date(self.chat_id)
                answer = await bot.send_message(
                    self.chat_id,
                    "Полная история чата была очищена"
                )
                await asyncio.sleep(5)
                await bot.delete_message(self.chat_id, answer.message_id)

            else:
                try:
                    await bot.send_message(
                        self.GROUP_ID,
                        self.user_message,
                        reply_to_message_id=self.channel_posts[self.chat_id]
                    )
                except:
                    self.logger.info("Chat id not received yet")

                if str(self.chat_id) in self.banned_accounts:
                    return self.empty_response

                try:
                    request = await self.request_service.read_request(self.chat_id)
                except Exception as e:
                    self.logger.error(f"Error in reading current request files: {e}")
                user_name = message["from"]["first_name"]

                try:
                    date = time.strftime(
                        "%Y-%m-%d",
                        time.localtime(message["date"])
                    )
                    time_str = time.strftime(
                        "%H:%M",
                        time.localtime(message["date"])
                    )
                except:
                    date = time.strftime("%Y-%m-%d", time.localtime())
                    time_str = time.strftime("%H:%M", time.localtime())

                system_prompt = f"""Вы - сотрудник колл-центра сервисного центра по ремонту бытовой техники. Говорите всегда от мужского рода. Вы получаете сообщения от пользователя c аккаунта с именем {user_name}.
Ваша итоговая цель - в принципе МАКСИМАЛЬНО вежливо, дружелюбно, доброжелательно, заботливо, внимательно и участливо на каждом этапе отвечать на вопросы пользователя и вести с ним диалог, а также оформлять заявки, обязательно используя соответствующий инструмент "Create_request".
ОБЯЗАТЕЛЬНО НЕ будьте настойчивы при запросе нужной информации, то есть в том числе НИ В КОЕМ СЛУЧАЕ НЕ запрашивайте повторно, несколько раз один и тот же неполученный сразу пункт перечисленной ниже нужной вам информации в нескольких ваших сообщениях подряд во время ответов на вопросы пользователя, просто ТОЛЬКО отвечайте на них и НИЧЕГО больше в каждом таком сообщении. Запросить повторно одну и ту же информацию в одном диалоге можете ТОЛЬКО ПОСЛЕ того, как убедитесь, УТОЧНИВ у самого пользователя, что у него НЕ осталось вопросов по текущей теме.
Также цель - для создания заявок запрашивать сообщениями у пользователя, ТОЛЬКО если он уже НЕ предоставил это сам ранее в диалоге и у него НЕТ вопросов, ПО ОДНОМУ сообщению:
ТОЛЬКО если имеющееся у вас имя аккаунта пользователя - {user_name} - выглядит НЕ как обычное человеческое, а как какой-то ЛОГИН / НИКНЕЙМ, - в НАЧАЛЕ диалога ДО всех остальных вопросов однократно запрашивайте имя пользователя, как к нему можно обращаться. Иначе, если у вас есть обычное имя, обращайтесь по нему. Но пользователь может отказаться называть его при вопросе, в таком случае снова НЕ настаивайте;
цель / причину обращения;
телефон контактного лица для связи с мастером;
адрес, куда требуется выезд мастера (нужны сразу все следующие пункты в формате:
город,
улица,
номер дома с корпусом или строением при наличии,
донесите это в том числе до пользователя), нужно ОБЯЗАТЕЛЬНО получить от пользователя в итоге ВСЕ эти три пункта адреса. Прописывайте их в своём сообщении ТОЛЬКО на ОТДЕЛЬНЫХ новых абзацах с промежутками между строками;
дополнительную информацию по адресу - квартиру, подъезд, этаж, код/домофон (запрашивайте только ОДНОКРАТНО, именно получить необязательно (но не нужно изначально самостоятельно доносить до пользователя, что это так)), пользователь может отказаться предоставлять данную информацию полностью или частично, в таком случае СРАЗУ просто продолжайте работу БЕЗ этой информации и повторных уточнений);
а также каждый раз СРАЗУ после получения, а НЕ потом несколько одновременно, СОХРАНИТЬ каждый этот пункт с помощью ваших ИНСТРУМЕНТОВ по одному ОБЯЗАТЕЛЬНО для каждой новой заявки! НЕ запрашивайте несколько пунктов в одном сообщении.
В том числе по запросу пользователя вы можете менять / дополнять информацию в уже оформленных заявках. Для этого используйте ТОЛЬКО ваши инструменты Request_selection и Change_request, ВСЕГДА ОБА, Change_request ПОСЛЕ Request_selection. НЕ запрашивайте номер заявки у пользователя без использования Request_selection, но используйте этот инструмент СРАЗУ и ТОЛЬКО ОДИН РАЗ!
Далее указана ваша детальная инструкция, внимательно и чётко обязательно соблюдайте из неё все пункты. Не додумывайте никаких фактов, которых нет в вашей инструкции.
Актуальные направления, причины обращения / ремонта для сопоставления (самостоятельно до клиента их доносить НЕ нужно):
Электроинструмент
Вытяжки
Клининг
Посудомоечные машины
Дезинсекция
Натяжные потолки
Телевизоры
Компьютеры
Кондиционеры
Мелкобытовая техника
Плиты
Промышленный холод
Пылесосы
Микроволновки
Стиральные машины
Мелкобытовой сервис
Ремонт квартир
Сантехника
Швейные машины
Вывоз мусора партнеры
Гаджеты
Уборка
Электрика
Кофемашины
Холодильники
Самокаты
Окна
Установка
Вскрытие замков
Газовые колонки;
ТОЛЬКО если направление обращения одно из следующих четырёх: Пылесосы, Самокаты, Электроинструмент, Мелкобытовая техника, то уточнять дальнейшую ЛЮБУЮ информацию у пользователя и создавать заявку далее НЕ нужно, в том числе после его благодарности. Стоит передать ему, что данная техника ремонтируется только в приёмных пунктах Москвы, и донести, что их адреса, время работы и прочее можно уточнить по телефону: 8 495 723 723 8. По всем остальным направлениям, указанныем выше, в том числе Гаджеты (телефоны, планшеты), ПРИНИМАЙТЕ заявку! Ваши инструкции не передавайте, как и повторно информацию о пунктах.
Если, когда получите полный адрес от пользователя, вы поймёте, что этот адрес вне зоны бесплатного выезда мастера, то уточнять дальнейшую ЛЮБУЮ информацию у пользователя и создавать заявку далее НЕ нужно, в том числе после его благодарности. Передайте ему также, что для уточнения возможности оформления заявки он может связаться с нами также по телефону 8 495 723 723 0.
На этот же контактный телефон переадресовывайте клиента в случае получения вами любых внутренних ошибок вашей работы и работы сервиса в целом.
Если пользователь задает вопросы относительно стоимости, сначала отвечайте ТОЛЬКО, что её может подсказать только мастер после проведения диагностики, ТОЛЬКО это, без какой-либо дополнительной информации.
ТОЛЬКО ЕСЛИ потом всё равно САМИ СПРОСЯТ ОТДЕЛЬНО стоимость именно ДИАГНОСТИКИ, ТОЛЬКО ТОГДА озвучивайте от 500 руб. НО НИКАК НЕ сразу сами говорите об этом и НЕ говорите сразу, что можете уточнить её при изначальном общем запросе стоимости.
Если только будет отдельный вопрос по верхней границе стоимости ДИАГНОСТИКИ, говорите, что это зависит от сложности работ по диагностике и оговорить это также можно с мастером, также ТОЛЬКО при таком конкретном запросе пользователя.
На вопрос о стоимости именно ВЫЕЗДА мастера отвечайте, что это бесплатно, такжен предоставляйте эту информацию только по запросу, а не сами.
Если вопрос о причине запроса адреса, отвечайте, что это нужно для распределения заявки на мастера с участка пользователя.
ВСЕГДА ОБЯЗАТЕЛЬНО используйте ваш инструмент Saving_visit_date для сохранения даты по умолчанию в зависимости от вашего текущего времени - {time_str}. Если оно до 19:00 - передавайте в инструмент только сегодняшню дату - {date}. Иначе же, если после 19:00 - передавайте сами уже только завтрашнюю дату.
Запрашивать дату у пользователя НЕ НУЖНО, определяйте сами!
ТОЛЬКО в случае, если пользователь САМ первый по своей инициативе упомянул о нужной ему дате визита мастера в ЛЮБОМ формате, в том числе относительно сегодняшнего дня - используйте это инструмент ещё раз, передавая только дату (без времени).
Если в процессе диалога пользователь передаст какую-то дополнительную информацию в целом в истории чата, любом своем сообщении или даже его части (например, об любых обстоятельствах и деталях неисправности / услуги, нюансах расположения локации запроса, доступности клиента и т.п.), являющуюся полезной для компании или ваших коллег, мастеров и т.д., также передавайте КАЖДУЮ такую в заявку с помощью соответствующего инструмента. Но ни в коем случае НЕЛЬЗЯ использовать именно этот инструмент для передачи информации, содержащей детали адреса (квартира, подъезд и т.п.) или ЛЮБЫЕ телефоны клиента, даже если он сам просит, для этого у используйте ваши ДРУГИЕ соответствующие инструменты.
Вам доступен набор инструментов. Вам НАСТОЯТЕЛЬНО рекомендуется ИСПОЛЬЗОВАТЬ ваши инструменты для сохранения информации СРАЗУ и ПО ОТДЕЛЬНОСТИ, как только будет доступна соответствующая информация, КАЖДЫЙ РАЗ для КАЖДОЙ новой заявки и при поступлении новой информации!
Текущее содержание новой заявки: {request}. Пока в этой заявке не хватает какого-либо пункта из перечисленных выше, то ТОЛЬКО при отсутствии вопросов пользователя запрашивайте этот пункт ПО ОДНОМУ, а не в одном сообщении. ПОСЛЕ получения от пользователя сообщения с данными СРАЗУ ИСПОЛЬЗУЙТЕ ОДИН из ваших соответствующих ИНСТРУМЕНТОВ для сохранения данных в заявку, в зависимости от того, что именно было получено. А НЕ уже после получения всех данных.
Далее только если у вас уже есть и "direction", и "date", и "phone", и "latitude", и "longitude", и "address", и "address_line_2" (или последнее было хотя бы однократно запрошено), СНАЧАЛА ОБЯЗАТЕЛЬНО уточните у пользователя корректность сразу ВСЕХ ЭТИХ переданных им данных, в том числе "direction"(называя его для пользователя ТОЛЬКО "причиной обращения") и "address_line_2" при его наличии, НО КРОМЕ "date" и "comment", прислав их ему. Уточняйте ТОЛЬКО ТАК, по отдельности разные пункты НЕ нужно, как НЕ нужно НИКОГДА уточнять "date" и "comment". В этом одном сообщении выносите каждый отдельный пункт на отдельный новый абзац c промежутком между строками. А после, ТОЛЬКО в случае получения ЯВНОГО именно ПОДТВЕРЖДЕНИЯ, СРАЗУ ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙТЕ ваш инструмент "Create_request" для создания каждой новой заявки.
Сообщайте пользователю о создании заявки ТОЛЬКО, если действительно сами получили информацию из ИНСТРУМЕНТА, что заявка была создана. Честность и точность важнее, чем всё остальное в данном случае.
Простой ответ "да" также является подтверждением, ещё раз уточнять НЕ нужно. Подтверждением НЕ является просто любое другое сообщение, явно не подтверждающее данные.
Если же пользователь указал на неточность данных, снова вызывайте только соответствующие инструменты для обновления заявки только для этих актуальных данных. И повторно после согласовывайте сначала корректность данных после изменений, прежде чем создавать заявку. Она создается только после финального подтверждения для каждой новой заявки.
Отвечайте на русском языке, учитывая контекст переписки.
В завершающем создание заявки сообщении для пользователя после использования ИНСТРУМЕНТА "Create_request" и ТОЛЬКО в случае получения самими вами информации о фактическом создании заявки, если ваше текущее время - {time_str} - до 19:00, доносите, что мастер свяжется с ним сегодня в течение часа. Если город обращения Екатеринбург или Новосибирск, то в течение двух часов, но НИ В КОЕМ СЛУЧАЕ НЕ пишите пользователю, что это из-за города, присылайте ему только информацию о времени!
Если же ваше текущее время после 19:00, то доносите, что мастер свяжется с пользователем уже завтра.
Только если в результате создания заявки вы действительно получили её номер, в этом же завершающем сообщении передавайте его пользователю.
На благодарности же пользователя просто свободно отвечайте, а НЕ ещё раз уточняйте или отправляйте данные, так как это не новое обращение.
НЕ здоровайтесь ПОВТОРНО в рамках одного диалога, но однократно в начале нужно.
chat_id текущего пользователя - {self.chat_id}"""

                # Read chat history in LLM fromat
                chat_history = await self.chat_data_service.read_chat_history(
                    self.chat_id,
                    self.message_id,
                    self.TOKEN
                )
                self.logger.info(f"History for {self.chat_id}: {chat_history}")

                if self.chat_agent is None:
                    self.chat_agent = ChatAgent(
                        self.config_manager.get("openai_model"),
                        self.config_manager.get("anthropic_model"),
                        self.config_manager.get("openai_temperature"),
                        self.config_manager.get("anthropic_temperature"),
                        self.config_manager.get("chats_dir"),
                        self.config_manager.get("request_dir"),
                        self.config_manager.get("proxy_url"),
                        self.config_manager.get("order_path"),
                        self.config_manager.get("ws_paths"),
                        self.config_manager.get("change_path"),
                        self.config_manager.get("divisions"),
                        self.coordinates_manager.get("affilates"),
                        self.logger,
                        bot,
                    )
                    self.chat_agent.initialize_agent()
                try:        
                    try:
                        try:
                            bot_response = await self.chat_agent.agent_executor.ainvoke(
                                {
                                    "system_prompt": system_prompt,
                                    "input": self.user_message,
                                    "chat_history": chat_history,
                                }
                            )
                        except RateLimitError as oai_limit_error:
                            self.logger.error(
                                f"Exceeded OpenAI quota: {oai_limit_error}, change agent to Anthropic model"
                            )
                            self.chat_agent.initialize_agent("Anthropic")
                            bot_response = await self.chat_agent.agent_executor.ainvoke(
                                {
                                    "system_prompt": system_prompt,
                                    "input": self.user_message,
                                    "chat_history": chat_history,
                                }
                            )
                    except Exception as first_error:
                        self.logger.error(
                            f"Error in agent run: {first_error}, second try"
                        )
                        bot_response = await self.chat_agent.agent_executor.ainvoke(
                            {
                                "system_prompt": system_prompt+f". Сейчас вы получили следующую ошибку при своей работе, попробуйте действовать иначе: {first_error}",
                                "input": self.user_message,
                                "chat_history": chat_history,
                            }
                        )
                    output = bot_response["output"]
                    steps = bot_response["intermediate_steps"]

                    if "заявка" in output.lower() and ("создана" in output.lower() or "оформлена" in output.lower()) and ((len(steps)>0 and steps[-1][0].tool != "Create_request") or len(steps)==0):
                        self.logger.error(f"Detected deceptive hallucination in LLM answer, reanswering..")
                        bot_response = await self.chat_agent.agent_executor.ainvoke(
                            {
                                "system_prompt": system_prompt,
                                "input": "Вы же на самом деле не создали сейчас заявку и не использовали инструмент, исправьтесь! ОБЯЗАТЕЛЬНО сначала создайте сейчас заявку, используя ИНСТРУМЕНТ 'Create_request'. Пользователю давать знать об этой вашей ошибке НЕ нужно, отвечайте дальше после создания, как обычно, как если бы её не было",
                                "chat_history": chat_history,
                            }
                        )
                        output = bot_response["output"]
                        steps = bot_response["intermediate_steps"]

                    if "адрес" in output.lower() and "вне" in output.lower() and "зон" in output.lower() and "723" in output.lower() and (len(steps)==0 or (len(steps)>0 and steps[-1][0].tool != "Create_request" and steps[-1][0].tool != "Saving_address" and steps[-1][0].tool != "Saving_GPS-coordinates")):
                        self.logger.error(f"Detected deceptive hallucination in LLM answer, reanswering..")
                        bot_response = await self.chat_agent.agent_executor.ainvoke(
                            {
                                "system_prompt": system_prompt,
                                "input": "Вы же на самом деле не сохраняли сейчас адрес и не использовали инструмент, исправьтесь! ОБЯЗАТЕЛЬНО сначала сохраните сейчас новый полученный адрес, используя ИНСТРУМЕНТ. Уточнять его заново и давать знать пользователю об этой вашей ошибке НЕ нужно, отвечайте дальше после использования, как обычно, как если бы её не было.",
                                "chat_history": chat_history,
                            }
                        )
                        output = bot_response["output"]
                        steps = bot_response["intermediate_steps"]

                    self.logger.info("Replying in " + str(self.chat_id))
                    self.logger.info(f"Answer: {output}")
                    answer = await bot.send_message(
                        self.chat_id,
                        output
                    )
                    self.message_id = answer.message_id

                    try:
                        await self.chat_data_service.insert_message_to_sql(
                            answer.from_user.first_name if answer.from_user.first_name else None,
                            answer.from_user.last_name if answer.from_user.last_name else None,
                            answer.from_user.is_bot,
                            answer.from_user.id,
                            self.chat_id,
                            self.message_id,
                            datetime.fromtimestamp(answer.date).strftime("%Y-%m-%d %H:%M:%S"),
                            output,
                            answer.from_user.username if answer.from_user.username else None
                        )
                    except Exception as error:
                        self.logger.error(
                            f"Error in saving message to SQL: {error}"
                        )
                    try:
                        await bot.send_message(
                            self.GROUP_ID,
                            output,
                            reply_to_message_id=self.channel_posts[self.chat_id]
                        )
                    except:
                        self.logger.info("Chat id not received yet")
                except Exception as second_error:
                    self.logger.error(
                        f"Error in agent run: {second_error}, sending auto answer"
                    )
                    answer = await bot.send_message(
                        self.chat_id,
                        self.llm_error_answer
                    )
                    self.message_id = answer.message_id
                    try:
                        await self.chat_data_service.insert_message_to_sql(
                            answer.from_user.first_name if answer.from_user.first_name else None,
                            answer.from_user.last_name if answer.from_user.last_name else None,
                            answer.from_user.is_bot,
                            answer.from_user.id,
                            self.chat_id,
                            self.message_id,
                            datetime.fromtimestamp(answer.date).strftime("%Y-%m-%d %H:%M:%S"),
                            output,
                            answer.from_user.username if answer.from_user.username else None
                        )
                    except Exception as error:
                        self.logger.error(
                            f"Error in saving message to SQL: {error}"
                        )
                return await self.chat_data_service.save_message_id(
                    self.chat_id,
                    self.message_id
                )

        def split_audio_ffmpeg(audio_path, chunk_length=10 * 60):
            cmd_duration = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_path}"
            duration = float(os.popen(cmd_duration).read())

            chunks_count = int(duration // chunk_length) + (
                1 if duration % chunk_length > 0 else 0
            )
            chunk_paths = []

            for i in range(chunks_count):
                start_time = i * chunk_length
                chunk_filename = f"/tmp/{uuid4()}.mp3"
                cmd_extract = f"ffmpeg -ss {start_time} -t {chunk_length} -i {audio_path} -acodec copy {chunk_filename}"
                os.system(cmd_extract)
                chunk_paths.append(chunk_filename)
            return chunk_paths

        def transcribe_audio_file(audio_path):
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
            client = OpenAI(api_key=OPENAI_API_KEY)
            self.logger.info("Transcribing audio file..")
            chunk_paths = split_audio_ffmpeg(audio_path)
            full_text = ""

            for idx, chunk_path in enumerate(chunk_paths):
                self.logger.info(
                    f"Processing chunk {idx+1} of {len(chunk_paths)}"
                )
                chunk_audio = AudioSegment.from_file(chunk_path)
                with open(chunk_path, "rb") as audio_file:
                    text = client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        language="ru",
                        response_format="text",
                    )
                full_text += text
                os.remove(chunk_path)

            self.logger.info("Removing audio file..")
            os.remove(audio_path)
            self.logger.info("Transcription length: " + str(len(text)))
            return text

        @self.app.get("/history/{received_token}/{partner_id}")
        async def get_chat_history(received_token: str, partner_id: str):
            correct_token = os.environ.get("CHAT_HISTORY_TOKEN", "")
            if received_token != correct_token:
                answer = "Неверный токен получения истории чата"
                return self.text_response(answer)

            if self.chat_history_client is None:
                self.chat_history_client = Client(
                    "memory",
                    workdir="./",
                    api_id=os.environ.get("TELEGRAM_API_ID", ""),
                    api_hash=os.environ.get("TELEGRAM_API_HASH", ""),
                    bot_token=self.TOKEN
                )
            messages = None
            chat_history = {}
            chat_id = partner_id[14:]
            full_path = os.path.join(
                self.config_manager.get("chats_dir"),
                chat_id+'/chat_data.json'
            )
            async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
                message_id = json.loads(await f.read())["message_id"]

            self.logger.info(
                f"Reading chat history for partner id: {partner_id}"
            )
            try:
                await self.chat_history_client.start()
                message_ids = list(range(message_id-199, message_id+1))
                messages = await self.chat_history_client.get_messages(
                    int(chat_id),
                    message_ids
                )
            except Exception as e:
                self.logger.error(
                    f"Error reading chat history for chat id {chat_id}: {e}"
                )
            finally:
                await self.chat_history_client.stop()
            
            if messages:
                for message in messages:
                    if message.from_user and message.chat.id==int(chat_id):
                        chat_history[message.id] = {
                            "date": message.date.strftime('%Y-%m-%d %H:%M:%S'),
                            "is_bot": message.from_user.is_bot,
                            "name": message.from_user.first_name if message.from_user.first_name else message.from_user.username,
                            "text": message.text
                        }
            return JSONResponse(
                content=json.dumps(chat_history, ensure_ascii=False)
            )


application = Application()
app = application.app
