from langchain.tools.base import StructuredTool
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
import os
import re
# from langchain.agents import initialize_agent, AgentType
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from geopy.geocoders import Nominatim
import requests
import json
from uuid import uuid4

from file_service import FileService
# from onec_request import OneC_Request


class save_direction_to_request_args(BaseModel):
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


class save_name_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    name: str = Field(description="name")


class create_request_args(BaseModel):
    name: str = Field(description="name")
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
    comment: str = Field(description="comment")
    name: str = Field(description="name")


class ChatAgent:

    def __init__(
        self,
        model,
        temperature,
        request_dir,
        order_url,
        get_url,
        clients,
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
            "request_dir": request_dir,
            "order_url": order_url,
            "get_url": get_url,
            "clients": clients,
        }
        self.agent_executor = None
        self.bot_instance = bot_instance

        self.request_service = FileService(self.config["request_dir"], logger)

    def initialize_agent(self):
        # llm = ChatAnthropic(
        #     api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=self.config["model"],
            temperature=self.config["temperature"],
        )
        tools = []

        # # Tool: get_directions_tool
        # get_directions_tool = StructuredTool.from_function(
        #     func=self.get_directions,
        #     name="Получение направлений",
        #     description="Предоставляет актуальный перечень направлений обращения / ремонта. ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙТЕ инструмент ОДИН раз в начале диалога ДО приветствия пользователя для возможности соотнесения последующей информации с этим перечнем.",
        #     return_direct=False,
        # )
        # tools.append(get_directions_tool)

        # Tool: save_direction_tool
        save_direction_tool = StructuredTool.from_function(
            coroutine=self.save_direction_to_request,
            name="Saving_direction",
            description="Сохраняет подходящее под запрос пользователя направление обращения из имеющегося списка направлений в заявку. Нужно соотнести запрос и выбрать подходящее только из тех, что в этом списке. Вам следует предоставить chat_id и непосредственно сам direction из списка в качестве параметров.",
            args_schema=save_direction_to_request_args,
            return_direct=False,
        )
        tools.append(save_direction_tool)

        # Tool: save_gps_tool
        save_gps_tool = StructuredTool.from_function(
            coroutine=self.save_gps_to_request,
            name="Saving_GPS-coordinates",
            description="Сохраняет адрес на основании полученнных GPS-координат в заявку. Вам следует предоставить значения chat_id, latitude и longitude в качестве параметров.",
            args_schema=save_gps_to_request_args,
            return_direct=False,
        )
        tools.append(save_gps_tool)

        # Tool: save_address_tool
        save_address_tool = StructuredTool.from_function(
            coroutine=self.save_address_to_request,
            name="Saving_address",
            description="Сохраняет полученнный адрес в заявку. Вам следует предоставить chat_id и непосредственно сам address из всего сообщения в качестве параметров.",
            args_schema=save_address_to_request_args,
            return_direct=False,
        )
        tools.append(save_address_tool)

        # Tool: save_address_line_2_tool
        save_address_line_2_tool = StructuredTool.from_function(
            coroutine=self.save_address_line_2_to_request,
            name="Saving_address_line_2",
            description="Сохраняет полученнную дополнительную информацию по адресу в заявку. Вам следует предоставить chat_id и непосредственно сам address_line_2 из всего сообщения в качестве параметов.",
            args_schema=save_address_line_2_to_request_args,
            return_direct=False,
        )
        tools.append(save_address_line_2_tool)

        # Tool: save_phone_tool
        save_phone_tool = StructuredTool.from_function(
            coroutine=self.save_phone_to_request,
            name="Saving_phone_number",
            description="Сохраняет полученнный телефон в заявку. Вам следует предоставить chat_id и непосредственно сам phone из всего сообщения в качестве параметов.",
            args_schema=save_phone_to_request_args,
            return_direct=False,
        )
        tools.append(save_phone_tool)

        # Tool: save_date_tool
        save_date_tool = StructuredTool.from_function(
            coroutine=self.save_date_to_request,
            name="Saving_visit_date",
            description="Сохраняет полученную дату визита в заявку. Вам следует самим предоставить в инструмент chat_id и непосредственно сам date в формате 'yyyy-mm-ddT00:00Z' из всего полученного сообщения в качестве параметров. ОЖИДАЙТЕ же и ПРИНИМАЙТЕ дату от пользователя в ЛЮБОМ свободном формате (например, 'сегодня' или 'завтра'), а НЕ в том, что выше. Главное используйте сами потом в указанном, отформатировав при необходимости.",
            args_schema=save_date_to_request_args,
            return_direct=False,
        )
        tools.append(save_date_tool)

        # Tool: save_comment_tool
        save_comment_tool = StructuredTool.from_function(
            coroutine=self.save_comment_to_request,
            name="Saving_comment",
            description="Сохраняет полезные по вашему мнению комментарии пользователя в заявку. Вам следует предоставить chat_id и непосредственно сам comment в качестве параметров.",
            args_schema=save_comment_to_request_args,
            return_direct=False,
        )
        tools.append(save_comment_tool)

        # Tool: save_name_tool
        save_name_tool = StructuredTool.from_function(
            coroutine=self.save_name_to_request,
            name="Saving_name",
            description="Сохраняет имя пользователя в заявку. Используйте, только если имя выглядит как настоящее человеческое. Вам следует предоставить chat_id и непосредственно само name в качестве параметров.",
            args_schema=save_name_to_request_args,
            return_direct=False,
        )
        tools.append(save_name_tool)

        # Tool: request_tool
        request_tool = StructuredTool.from_function(
            func=self.create_request,
            name="Request_creation",
            description="Создает полностью заполненную заявку в 1С и по возможности определяет её номер. Вам следует предоставить по отдельности сами значения ключей словаря (request) с текущей заявкой из вашего системного промпта в качестве соответствующих параметров инструмента, кроме ключа address_line_2. Из его же значения выделите и передайте отдельно при наличии непосредственно сами численно-буквенные значения apartment, entrance, floor и intercom (т.е. без слов) из всего address_line_2 в качестве остальных соответствующих параметров инструмента. Если какие-то из параметров не были предоставлены пользователем, передавайте их в инструмент со значением пустой строки - ''",
            args_schema=create_request_args,
            return_direct=False,
        )
        tools.append(request_tool)

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
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # def get_directions(self):
    #     with open("./data/repair_dir.txt", "r", encoding="utf-8") as f:
    #         directions = f.read().splitlines()
    #     directions_string = ", ".join(directions)
    #     return f"Актуальные направления обращения / ремонта для сопоставления: {directions_string}"

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
        location = geolocator.geocode(address)
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

    async def save_name_to_request(self, chat_id, name):
        self.logger.info(f"save_name_to_request name: {name}")
        await self.request_service.save_to_request(chat_id, name, "name")
        self.logger.info("Имя пользователя было сохранено в заявку")
        return "Имя пользователя было сохранено в заявку"

    def create_request(
        self,
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
        comment="",
        name="Имя",
    ):
        token = os.environ.get("1С_TOKEN", "")
        login = os.environ.get("1C_LOGIN", "")
        password = os.environ.get("1C_PASSWORD", "")

        with open("./data/template.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        params["order"]["client"]["display_name"] = name
        params["order"]["uslugi_id"] = str(uuid4()).replace("-", "")

        params["order"]["services"][0]["service_id"] = direction
        params["order"]["desired_dt"] = date
        params["order"]["client"]["phone"] = phone
        params["order"]["address"]["geopoint"]["latitude"] = latitude
        params["order"]["address"]["geopoint"]["longitude"] = longitude
        params["order"]["address"]["name"] = address
        params["order"]["address"]["floor"] = floor
        params["order"]["address"]["entrance"] = entrance
        params["order"]["address"]["apartment"] = apartment
        params["order"]["address"]["intercom"] = intercom
        params["order"]["comment"] = comment
        self.logger.info(f"Parametrs: {params}")

        request_data = {"token": token, "params": params}
        order_url = f"{self.config['order_url']}/create_order"
        get_url = f"{self.config['get_url']}/ws"

        order = requests.post(
            order_url, json=request_data, headers={"Content-Type": "application/json"}
        )
        self.logger.info(f"Result:\n{order.status_code}\n{order.text}")

        query_params = {
            "Идентификатор": "bid_info",
            "НомерПартнера": params["order"]["uslugi_id"],
        }
        config_data = {
            "clientPath": self.config["clients"],
            "login": login,
            "password": password,
        }
        # query_params = json.dumps(query_params, ensure_ascii=False)
        # onec_request = OneC_Request(login, password, self.config["clients"])

        request_number = None
        try:
            results = requests.post(
                get_url, json={"config": config_data, "query_params": query_params}
            ).json()
            for value in results.values():
                if len(value) > 0:
                    request_number = value[0]["id"]
                    break
        except Exception as e:
            self.logger.error(f"Error in receiving request number: {e}")

        if order.status_code == 200:
            self.logger.info(f"number: {request_number}")
            if request_number:
                return f"Заявка была создана с номером {request_number}"
            else:
                return "Заявка была создана"
        else:
            return f"Ошибка при создании заявки: {order.text}"
