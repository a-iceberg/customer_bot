import os
import re
import time
import json
import requests

from datetime import datetime
from openai import OpenAI
from pydantic.v1 import BaseModel, Field
from geopy.geocoders import Nominatim
from telebot.types import ReplyKeyboardMarkup

from langchain_core.tools import StructuredTool
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from file_service import FileService


class save_name_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    name: str = Field(description="name")


class SaveDirectionToRequestArgs(BaseModel):
    chat_id: int = Field(description="chat_id")
    direction: str = Field(description="direction")


class save_gps_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    latitude: float = Field(description="latitude")
    longitude: float = Field(description="longitude")


class save_address_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    address: str = Field(description="address")


class save_address_line_2_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    address_line_2: str = Field(description="address_line_2")


class save_phone_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    phone: str = Field(description="phone")


class save_date_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    date: str = Field(description="date")


class save_comment_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    comment: str = Field(description="comment")


class create_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    city: str = Field(description="city")
    direction: str = Field(description="direction")
    date: str = Field(description="date")
    phone: str = Field(description="phone")
    latitude: float = Field(description="latitude")
    longitude: float = Field(description="longitude")
    address: str = Field(description="address")
    apartment: str = Field(description="apartment")
    entrance: str = Field(description="entrance")
    floor: str = Field(description="floor")
    intercom: str = Field(description="intercom")
    name: str = Field(description="name")
    comment: str = Field(description="comment")


class request_selection_args(BaseModel):
    chat_id: int = Field(description="chat_id")


class change_request_args(BaseModel):
    request_number: str = Field(description="request_number")
    field_name: str = Field(description="field_name")
    field_value: str = Field(description="field_value")


class ChatAgent:

    def __init__(
        self,
        model,
        temperature,
        chats_dir,
        request_dir,
        proxy_url,
        order_path,
        ws_paths,
        change_path,
        logger,
        bot_instance,
    ):
        self.logger = logger
        self.logger.info(
            f"ChatAgent init with model: {model} and temperature: {temperature}"
        )
        self.config = {
            "model": model,
            "temperature": temperature,
            "chats_dir": chats_dir,
            "request_dir": request_dir,
            "proxy_url": proxy_url,
            "order_path": order_path,
            "ws_paths": ws_paths,
            "change_path": change_path
        }
        self.agent_executor = None
        self.bot_instance = bot_instance

        self.request_service = FileService(self.config["request_dir"], self.logger)
        self.chat_history_service = FileService(self.config["chats_dir"], self.logger)


    def initialize_agent(self):
        # llm = ChatAnthropic(
        #     api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=self.config["model"],
            temperature=self.config["temperature"],
        )
        tools = []

        # Tool: save_name_tool
        save_name_tool = StructuredTool.from_function(
            coroutine=self.save_name_to_request,
            name="Saving_name",
            description="Сохраняет имя пользователя в заявку. Используйте этот инструмент ОБЯЗАТЕЛЬНО ВСЕГДА, если имеющееся у вас или полученное имя выглядит как настоящее человеческое. Вам следует предоставить chat_id и непосредственно само name в качестве параметров.",
            args_schema=save_name_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_name_tool)

        # Tool: save_direction_tool
        save_direction_tool = StructuredTool.from_function(
            coroutine=self.save_direction_to_request,
            name="Saving_direction",
            description="Сохраняет подходящее под запрос пользователя направление обращения из имеющегося списка направлений в заявку. Нужно соотнести запрос и выбрать подходящее только из тех, что в этом списке. Вам следует предоставить chat_id и непосредственно сам direction из списка в качестве параметров.",
            args_schema=SaveDirectionToRequestArgs,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_direction_tool)

        # Tool: save_gps_tool
        save_gps_tool = StructuredTool.from_function(
            coroutine=self.save_gps_to_request,
            name="Saving_GPS-coordinates",
            description="Сохраняет адрес на основании полученнных GPS-координат в заявку. Вам следует предоставить значения chat_id, latitude и longitude в качестве параметров.",
            args_schema=save_gps_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_gps_tool)

        # Tool: save_address_tool
        save_address_tool = StructuredTool.from_function(
            coroutine=self.save_address_to_request,
            name="Saving_address",
            description="Сохраняет полученнный адрес в заявку. При сохранении убедитесь, что у вас есть ВСЕ три обязательных поля адреса (город, улица, дом). Вам следует предоставить chat_id и непосредственно сам address из всего сообщения в качестве параметров.",
            args_schema=save_address_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_address_tool)

        # Tool: save_address_line_2_tool
        save_address_line_2_tool = StructuredTool.from_function(
            coroutine=self.save_address_line_2_to_request,
            name="Saving_address_line_2",
            description="Сохраняет полученнную дополнительную информацию по адресу в заявку. Вам следует предоставить chat_id и непосредственно сам address_line_2 из всего сообщения в качестве параметов.",
            args_schema=save_address_line_2_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_address_line_2_tool)

        # Tool: save_phone_tool
        save_phone_tool = StructuredTool.from_function(
            coroutine=self.save_phone_to_request,
            name="Saving_phone_number",
            description="Сохраняет полученнный телефон в заявку. Вам следует предоставить chat_id и непосредственно сам phone из всего сообщения в качестве параметов.",
            args_schema=save_phone_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_phone_tool)

        # Tool: save_date_tool
        save_date_tool = StructuredTool.from_function(
            coroutine=self.save_date_to_request,
            name="Saving_visit_date",
            description="Сохраняет нужную дату визита в заявку. Вам следует САМИМ предоставить в инструмент chat_id и непосредственно саму date в формате 'yyyy-mm-ddT00:00Z', определённую вами самостоятельно по умолчанию или же полученную из сообщения пользователя в качестве параметров. ПРИНИМАЙТЕ дату от пользователя в ЛЮБОМ свободном формате (например, 'сегодня' или 'завтра'), а НЕ в том, что выше. Главное используйте сами потом в указанном, отформатировав при необходимости.",
            args_schema=save_date_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_date_tool)

        # Tool: save_comment_tool
        save_comment_tool = StructuredTool.from_function(
            coroutine=self.save_comment_to_request,
            name="Saving_comment",
            description="Сохраняет полезные по вашему мнению комментарии пользователя в заявку. Ни в коем случае НЕЛЬЗЯ передавать здесь информацию, содержающую детали адреса (квартира, подъезд и т.п.) или ЛЮБЫЕ телефоны клиента, даже если он просит, в таком случае НЕ используйте этот инструмент. Вам следует предоставить chat_id и непосредственно сам comment в качестве параметров.",
            args_schema=save_comment_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_comment_tool)

        # Tool: create_request_tool
        create_request_tool = StructuredTool.from_function(
            func=self.create_request,
            name="Create_request",
            description="Создает полностью заполненную заявку в 1С и по возможности определяет её номер. Вам следует предоставить chat_id, точное значение города обращения из списка в вашем системном промпте, выбранное на основании города в address, и по отдельности сами значения ключей словаря (request) с текущей заявкой из вашего системного промпта в качестве соответствующих параметров инструмента, кроме ключа address_line_2. Из его же значения выделите и передайте отдельно при наличии непосредственно сами численно-буквенные значения apartment, entrance, floor и intercom (т.е. без слов) из всего address_line_2 в качестве остальных соответствующих параметров инструмента.",
            args_schema=create_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(create_request_tool)

        # Tool: request_selection_tool
        request_selection_tool = StructuredTool.from_function(
            coroutine=self.request_selection,
            name="Request_selection",
            description="Находит и предоставляет пользователю список его текущих заявок для выбора, чтобы определить контекст всего диалога, если речь идёт уже о каких-либо созданных заявках, а не об оформлении новой. Используйте этот инструмент ОБЯЗАТЕЛЬНО, когда спрашивайте номер заявки у пользователя, но ТОЛЬКО ОДИН РАЗ, когда вам нужно понять, о какой именно заявке идёт речь, например, СРАЗУ, как только пользователь захочет изменить или дополнить данные по уже существующей заявке. Если вы уже явно получили от пользователя номер заявки, повторно НИ В КОЕМ СЛУЧАЕ НЕ используйте этот инструмент! Вам следует предоставить chat_id в качестве параметра.",
            args_schema=request_selection_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(request_selection_tool)

        # Tool: change_request_tool
        change_request_tool = StructuredTool.from_function(
            func=self.change_request,
            name="Change_request",
            description="Изменяет нужные данные / значения полей в уже существующей заявке. Допустимо обрабатывать ТОЛЬКО комментарий или телефон. Для редактирования уже имеющихся созданных заявок используйте ТОЛЬКО ЭТОТ инструмент, а НЕ обычные с добавлением информации в новую! Вам следует предоставить сам номер текущей заявки request_number; field_name - подходящее название поля: 'comment' или 'phone'; а также само новое значение поля, полученное от пользователя (field_value) в качестве параметров.",
            args_schema=change_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(change_request_tool)

        # self.agent = initialize_agent(
        #     tools,
        #     llm,
        #     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        #     verbose=True,
        #     handle_parsing_errors=True,
        # )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, early_stopping_method="generate", max_iterations=20
        )

    async def save_name_to_request(self, chat_id, name):
        self.logger.info(f"save_name_to_request name: {name}")
        await self.request_service.save_to_request(chat_id, name, "name")
        self.logger.info("Имя пользователя было сохранено в заявку")
        return "Имя пользователя было сохранено в заявку"

    async def save_direction_to_request(self, chat_id, direction):
        self.logger.info(f"save_direction_to_request direction: {direction}")
        await self.request_service.save_to_request(chat_id, direction, "direction")
        self.logger.info("Направление обращения было сохранено в заявку")
        return "Направление обращения было сохранено в заявку"

    async def save_gps_to_request(self, chat_id, latitude, longitude):
        self.logger.info(
            f"save_gps_to_request latitude: {latitude} longitude: {longitude}"
        )
        geolocator = Nominatim(user_agent="my_app")
        address = geolocator.reverse(f"{latitude}, {longitude}").address
        await self.request_service.save_to_request(chat_id, latitude, "latitude")
        await self.request_service.save_to_request(chat_id, longitude, "longitude")
        await self.request_service.save_to_request(chat_id, address, "address")
        self.logger.info("Адрес пользователя был сохранен в заявку")
        return "Адрес пользователя был сохранен в заявку"

    async def save_address_to_request(self, chat_id, address):
        self.logger.info(f"save_address_to_request address: {address}")
        geolocator = Nominatim(user_agent="my_app")
        location = None
        try:
            location = geolocator.geocode(address)
        except:
            self.logger.error(f"Failed to get coordinates for {address}")
        if location:
            latitude = location.latitude
            longitude = location.longitude
        else:
            latitude = 55.900678
            longitude = 37.528109
        await self.request_service.save_to_request(chat_id, latitude, "latitude")
        await self.request_service.save_to_request(chat_id, longitude, "longitude")
        await self.request_service.save_to_request(chat_id, address, "address")
        self.logger.info("Адрес пользователя был сохранен в заявку")
        return "Адрес пользователя был сохранен в заявку"

    async def save_address_line_2_to_request(self, chat_id, address_line_2):
        self.logger.info(f"save_address_line_2_to_request address: {address_line_2}")
        await self.request_service.save_to_request(
            chat_id, address_line_2, "address_line_2"
        )
        self.logger.info("Вторая линия адреса пользователя была сохранена в заявку")
        return "Вторая линия адреса пользователя была сохранена в заявку"

    async def save_phone_to_request(self, chat_id, phone):
        self.logger.info(f"save_phone_to_request phone: {phone}")
        await self.request_service.save_to_request(
            chat_id, "".join(re.findall(r"[\d]", phone)), "phone"
        )
        self.logger.info("Телефон пользователя был сохранен в заявку")
        return "Телефон пользователя был сохранен в заявку"

    async def save_date_to_request(self, chat_id, date):
        self.logger.info(f"save_date_to_request date: {date}")
        await self.request_service.save_to_request(chat_id, date, "date")
        self.logger.info("Дата посещения была сохранена в заявку")
        return "Дата посещения была сохранена в заявку"

    async def save_comment_to_request(self, chat_id, comment):
        self.logger.info(f"save_comment_to_request comment: {comment}")
        await self.request_service.save_to_request(chat_id, comment, "comment")
        self.logger.info("Комментарий был сохранен в заявку")
        return "Комментарий был сохранен в заявку"


    def create_request(
        self,
        chat_id,
        city,
        direction,
        date,
        phone,
        latitude,
        longitude,
        address,
        apartment="",
        entrance="",
        floor="",
        intercom="",
        name="Не названо",
        comment=""
    ):
        token = os.environ.get("1С_TOKEN", "")
        login = os.environ.get("1C_LOGIN", "")
        password = os.environ.get("1C_PASSWORD", "")

        with open("./data/template.json", "r", encoding="utf-8") as f:
            order_params = json.load(f)

        order_params["order"]["uslugi_id"] = str(chat_id)+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()).replace("-", "")
        order_params["order"]["client"]["display_name"] = name
        order_params["order"]["services"][0]["service_id"] = direction
        order_params["order"]["desired_dt"] = date
        order_params["order"]["client"]["phone"] = phone
        order_params["order"]["address"]["name"] = address
        order_params["order"]["address"]["floor"] = floor
        order_params["order"]["address"]["entrance"] = entrance
        order_params["order"]["address"]["apartment"] = apartment
        order_params["order"]["address"]["intercom"] = intercom
        order_params["order"]["address"]["name_components"][0]["name"] = city

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        response = client.chat.completions.create(
            model=self.config["model"],
            temperature=0,
            seed=654321,
            messages=[
                {
                    "role": "system",
                    "content": "Вы - сотрудник по сохранности конфиденциальных данных. В передаваемом вами тексте никогда не должно быть никакой следующей информации: любых номеров телефонов; значений подъезда, этажа, квартиры, домофона. Возвращайте в ответе ТОЛЬКО полученный текст с УБРАННОЙ всей перечисленной выше информацией, НИ В КОЕМ СЛУЧАЕ НЕ ваш ответ с размышлениями. Если текст изначально пустой, также возвращайте пустую строку - ''.",
                },
                {
                    "role": "user",
                    "content": "проход под аркой домофон 45к7809в, этаж 10, квартира 45, подъезд 3, дополнительный телефон 89760932378",
                },
                {
                    "role": "assistant",
                    "content": "проход под аркой домофон, этаж, квартира, подъезд, дополнительный телефон",
                },
                {"role": "user", "content": comment},
            ]
        )
        comment = response.choices[0].message.content
        pattern = re.compile(r"([+]?[\d]?\d{3}.*?\d{3}.*?\d{2}.*?\d{2})|подъезд|этаж|эт|квартир|кв|домофон|код", re.IGNORECASE)
        comment = re.sub(pattern, '', comment)

        if latitude == 0:
            latitude = 55.760221
        if longitude == 0:
            longitude = 37.618561

        order_params["order"]["comment"] = comment
        order_params["order"]["address"]["geopoint"]["latitude"] = latitude
        order_params["order"]["address"]["geopoint"]["longitude"] = longitude        
        self.logger.info(f"Parametrs: {order_params}")

        ws_params = {
            "Идентификатор": "new_bid_number",
            "НомерПартнера": order_params["order"]["uslugi_id"],
        }
        order_url = f"{self.config['proxy_url']}/hs"
        ws_url = f"{self.config['proxy_url']}/ws"

        order_data = {
            "clientPath": self.config["order_path"]
        }
        order = requests.post(
            order_url,
            json={"config": order_data, "params": order_params, "token": token},
        )
        self.logger.info(f"Result:\n{order.status_code}\n{order.text}")

        ws_data = {
            "clientPath": self.config["ws_paths"],
            "login": login,
            "password": password,
        }
        request_number = None
        try:
            results = requests.post(
                ws_url, json={"config": ws_data, "params": ws_params, "token": token}
            ).json()["result"]
            self.logger.info(f"results: {results}")
            for value in results.values():
                if len(value) > 0:
                    request_number = value[0]["id"]
                    break
        except Exception as e:
            self.logger.error(f"Error in receiving request number: {e}")

        if order.status_code == 200:
            self.logger.info(f"number: {request_number}")
            self.request_service.delete_files(chat_id)
            if request_number:
                return f"Заявка была создана с номером {request_number}"
            else:
                return "Заявка была создана"
        else:
            return f"Ошибка при создании заявки: {order.text}"

    async def request_selection(self, chat_id):
        token = os.environ.get("1С_TOKEN", "")
        login = os.environ.get("1C_LOGIN", "")
        password = os.environ.get("1C_PASSWORD", "")

        ws_url = f"{self.config['proxy_url']}/ws"        
        ws_params = {
            "Идентификатор": "bid_numbers",
            "НомерПартнера": str(chat_id),
        }
        ws_data = {
            "clientPath": self.config["ws_paths"],
            "login": login,
            "password": password,
        }
        request_numbers = []
        try:
            results = requests.post(
                ws_url, json={"config": ws_data, "params": ws_params, "token": token}
            ).json()["result"]
            self.logger.info(f"results: {results}")
            for value in results.values():
                if len(value) > 0:
                    for request in value:
                        request_numbers.append(request["id"])
            self.logger.info(f"request_numbers: {request_numbers}")
        except Exception as e:
            self.logger.error(f"Error in receiving request numbers: {e}")

        if len(request_numbers) > 0:
            markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            text = "Секунду..."
            for number in sorted(request_numbers):
                markup.add(f"Номер моей заявки - {number}")
            markup.add("🏠 Вернуться в меню")
            self.bot_instance.send_message(chat_id, text, reply_markup=markup)

            return "У пользователя был только запрошен номер заявки, в рамках которой сейчас идёт диалог"
        else:
            return "У пользователя нет существующих заявок"
    
    def change_request(self, request_number, field_name, field_value):
        token = os.environ.get("1С_TOKEN", "")
        login = os.environ.get("1C_LOGIN", "")
        password = os.environ.get("1C_PASSWORD", "")

        partner_number = None
        date_str = None
        revision = None

        ws_url = f"{self.config['proxy_url']}/ws"        
        ws_params = {
            "Идентификатор": "data_to_change_bid",
            "Номер": request_number,
        }
        ws_data = {
            "clientPath": self.config["ws_paths"],
            "login": login,
            "password": password,
        }
        try:
            results = requests.post(
                ws_url, json={"config": ws_data, "params": ws_params, "token": token}
            ).json()["result"]
            self.logger.info(f"results: {results}")
            for value in results.values():
                if len(value) > 0:
                    partner_number = str(value[0]["id"])
                    date_received = value[0]["date"]
                    date = datetime.strptime(date_received, '%d.%m.%Y %H:%M:%S')
                    date_str = date.strftime('%Y-%m-%dT%H:%MZ')
                    if value[0]["comment"]:
                        if value[0]["comment"] == "''":
                            comment = ''
                        else:
                            comment = value[0]["comment"]
                    else:
                        comment = ''
                    # comment = value[0]["comment"] if value[0]["comment"] else ''
                    break

            get_url = f"{self.config['proxy_url']}/rev"
            get_data = {
                "clientPath": {"crm": self.config["order_path"]["crm"]+partner_number}
            }
            request = requests.post(
                get_url, json={"config": get_data, "token": token}
            ).json()["result"]["order"]
            revision = request["revision"]
            locality = request["address"]["name_components"][0]["name"]

            self.logger.info(f"partner_number: {partner_number}")
            self.logger.info(f"date: {date_str}")
            self.logger.info(f"comment: {comment}")
            self.logger.info(f"revision: {revision}")
            self.logger.info(f"locality: {locality}")
        except Exception as e:
            self.logger.error(f"Error in receiving request data: {e}")

        if partner_number and date_str and locality and revision is not None:
            with open("./data/template.json", "r", encoding="utf-8") as f:
                change_params = json.load(f)
            change_params["order"]["uslugi_id"] = partner_number
            change_params["order"]["desired_dt"] = date_str
            change_params["order"]["address"]["name_components"][0]["name"] = locality
            change_params["order"]["revision"] = revision + 1
            if field_name == "comment":
                change_params["order"]["comment"] = field_value
                self.logger.info(f"Parametrs: {change_params}")
            elif field_name == "phone":
                change_params["order"]["client"]["phone"] = field_value
                change_params["order"]["comment"] = comment
                self.logger.info(f"Parametrs: {change_params}")
            else:
                self.logger.info(f"Parametrs: {change_params}")
                return "Получено или сформулировано недопустимое для изменения значение. Доступны только коммментарий или телефон"
            
            change_url = f"{self.config['proxy_url']}/ex"
            change_data = {
                "clientPath": {"crm": self.config["change_path"]["crm"]+partner_number}
            }
            change = requests.post(
                change_url,
                json={"config": change_data, "params": change_params, "token": token}
            )
            self.logger.info(f"Result:\n{change.status_code}\n{change.text}")
            if change.status_code == 200:
                return f"Данные заявки были обновлнены"
            else:
                return f"Ошибка при обновлении заявки: {change.text}"
        else:
            return "Произошла ошибка при получении данных заявки"