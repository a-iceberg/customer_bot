from langchain.tools.base import StructuredTool
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
import os
import re
from langchain.agents import initialize_agent, AgentType
from geopy.geocoders import Nominatim
import requests
import json
from uuid import uuid4

from file_service import FileService


class save_direction_to_request_args(BaseModel):
    direction: str = Field(description="appeal_direction")


class save_gps_to_request_args(BaseModel):
    latitude: float = Field(description="latitude")
    longitude: float = Field(description="longitude")


class save_address_to_request_args(BaseModel):
    address: str = Field(description="address")


class save_phone_to_request_args(BaseModel):
    phone: str = Field(description="phone")


class save_date_to_request_args(BaseModel):
    date: str = Field(description="date")


class save_model_to_request_args(BaseModel):
    model: str = Field(description="brand and model")


class save_circs_to_request_args(BaseModel):
    circs: str = Field(description="circumstances")


class create_request_args(BaseModel):
    direction: str = Field(description="appeal_direction")
    latitude: float = Field(description="latitude")
    longitude: float = Field(description="longitude")
    address: str = Field(description="address")
    phone: str = Field(description="phone")
    date: str = Field(description="date")
    comment: str = Field(description="comment")


class ChatAgent:
    def __init__(
        self, model, temperature, request_dir, base_url, logger, bot_instance, chat_id
    ):
        self.logger = logger
        self.logger.info(
            f"ChatAgent init with model: {model} and temperature: {temperature}"
        )
        self.config = {
            "model": model,
            "temperature": temperature,
            "request_dir": request_dir,
            "base_url": base_url,
        }
        self.agent = None
        self.bot_instance = bot_instance
        self.chat_id = chat_id

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

        # Tool: get_directions_tool
        get_directions_tool = StructuredTool.from_function(
            func=self.get_directions,
            name="Получение направлений",
            description="Предоставляет актуальный перечень направлений обращения / ремонта. Обязательно используйте после получения от пользователя причины, направления обращения и ДО использования инструмента сохранения направления для последующего соотнесения полученной информации с этим перечнем.",
            return_direct=False,
        )
        tools.append(get_directions_tool)

        # Tool: save_direction_tool
        save_direction_tool = StructuredTool.from_function(
            coroutine=self.save_direction_to_request,
            name="Сохранение направления обращения",
            description="Сохраняет подходящее под запрос пользователя направление обращения из имеющегося списка направлений в заявку. Нужно соотнести запрос и выбрать подходящее только из тех, что в этом списке. Если подходящее направление одно из следующих: Пылесосы, Самокаты, Электроинструмент, Мелкобытовая техника, то уточнять дальнейшую информацию у пользователя и создавать заявку далее не нужно. Стоит передать ему, что данная техника ремонтируется только в приёмных пунктах, чьи адреса, время работы и прочее можно уточнить по телефону: 8 495 723 723 8. Вам следует предоставить только непосредственно сам direction из списка в качестве параметра.",
            args_schema=save_direction_to_request_args,
            return_direct=False,
        )
        tools.append(save_direction_tool)

        # Tool: save_gps_tool
        save_gps_tool = StructuredTool.from_function(
            coroutine=self.save_gps_to_request,
            name="Сохранение GPS-координат",
            description="Сохраняет адрес на основании полученнных GPS-координат в заявку. Вам следует предоставить значения latitude и longitude в качестве параметров.",
            args_schema=save_gps_to_request_args,
            return_direct=False,
        )
        tools.append(save_gps_tool)

        # Tool: save_address_tool
        save_address_tool = StructuredTool.from_function(
            coroutine=self.save_address_to_request,
            name="Сохранение адреса",
            description="Сохраняет полученнный адрес в заявку. Вам следует предоставить только непосредственно сам address из всего сообщения в качестве параметра.",
            args_schema=save_address_to_request_args,
            return_direct=False,
        )
        tools.append(save_address_tool)

        # Tool: save_phone_tool
        save_phone_tool = StructuredTool.from_function(
            coroutine=self.save_phone_to_request,
            name="Сохранение телефона",
            description="Сохраняет полученнный телефон в заявку. Вам следует предоставить только непосредственно сам phone из всего сообщения в качестве параметра.",
            args_schema=save_phone_to_request_args,
            return_direct=False,
        )
        tools.append(save_phone_tool)

        # Tool: save_date_tool
        save_date_tool = StructuredTool.from_function(
            coroutine=self.save_date_to_request,
            name="Сохранение даты визита",
            description="Сохраняет желаемую дату визита в заявку. Вам следует предоставить только непосредственно сам date в строгом формате 'yyyy-mm-ddT00:00Z' из всего сообщения в качестве параметра. Запрашивать же дату у пользователя можно в свободном формате, главное используйте сами в указанном.",
            args_schema=save_date_to_request_args,
            return_direct=False,
        )
        tools.append(save_date_tool)

        # Tool: save_model_tool
        save_model_tool = StructuredTool.from_function(
            coroutine=self.save_model_to_request,
            name="Сохранение бренда и модели",
            description="Сохраняет бренд и модель техники в заявку. Вам следует предоставить только непосредственно сам model, содержащий и бренд, и модель, из всего сообщения в качестве одного параметра.",
            args_schema=save_model_to_request_args,
            return_direct=False,
        )
        tools.append(save_model_tool)

        # Tool: save_circs_tool
        save_circs_tool = StructuredTool.from_function(
            coroutine=self.save_circs_to_request,
            name="Сохранение обстоятельств обращения",
            description="Сохраняет дополнительные полезные обстоятельства обращения в заявку. Вам следует предоставить только непосредственно сами circs из всего сообщения в качестве параметра.",
            args_schema=save_circs_to_request_args,
            return_direct=False,
        )
        tools.append(save_circs_tool)

        # Tool: request_tool
        request_tool = StructuredTool.from_function(
            func=self.create_request,
            name="Создание заявки",
            description="Создает полностью заполненную заявку в 1С. Вам следует предоставить значения ключей request в качестве параметров.",
            args_schema=create_request_args,
            return_direct=False,
        )
        tools.append(request_tool)

        self.agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )

    def get_directions(self):
        with open("./data/repair_dir.txt", "r", encoding="utf-8") as f:
            directions = f.read().splitlines()
        directions_string = ", ".join(directions)
        return f"Актуальные направления обращения / ремонта для сопоставления: {directions_string}"

    async def save_direction_to_request(self, direction):
        self.logger.info(f"save_direction_to_request direction: {direction}")
        await self.request_service.save_to_request(self.chat_id, direction, "direction")
        self.logger.info("Направление обращения было сохранено в заявку")
        return "Направление обращения было сохранено в заявку"

    async def save_gps_to_request(self, latitude, longitude):
        self.logger.info(
            f"save_gps_to_request latitude: {latitude} longitude: {longitude}"
        )
        geolocator = Nominatim(user_agent="my_app")
        address = geolocator.reverse(f"{latitude}, {longitude}").address
        await self.request_service.save_to_request(self.chat_id, latitude, "latitude")
        await self.request_service.save_to_request(self.chat_id, longitude, "longitude")
        await self.request_service.save_to_request(self.chat_id, address, "address")
        self.logger.info("Адрес пользователя был сохранен в заявку")
        return "Адрес пользователя был сохранен в заявку"

    async def save_address_to_request(self, address):
        self.logger.info(f"save_address_to_request address: {address}")
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.geocode(address)
        if location:
            latitude = location.latitude
            longitude = location.longitude
        else:
            latitude = 55.900678
            longitude = 37.528109
        await self.request_service.save_to_request(self.chat_id, latitude, "latitude")
        await self.request_service.save_to_request(self.chat_id, longitude, "longitude")
        await self.request_service.save_to_request(self.chat_id, address, "address")
        self.logger.info("Адрес пользователя был сохранен в заявку")
        return "Адрес пользователя был сохранен в заявку"

    async def save_phone_to_request(self, phone):
        self.logger.info(f"save_phone_to_request phone: {phone}")
        await self.request_service.save_to_request(
            self.chat_id, "".join(re.findall(r"[\d]", phone)), "phone"
        )
        self.logger.info("Телефон пользователя был сохранен в заявку")
        return "Телефон пользователя был сохранен в заявку"

    async def save_date_to_request(self, date):
        self.logger.info(f"save_date_to_request date: {date}")
        await self.request_service.save_to_request(self.chat_id, date, "date")
        self.logger.info("Дата посещения была сохранена в заявку")
        return "Дата посещения была сохранена в заявку"

    async def save_model_to_request(self, model):
        self.logger.info(f"save_model_to_request model: {model}")
        await self.request_service.save_to_request(
            self.chat_id, f"Модель: {model}", "comment"
        )
        self.logger.info("Бренд и модель техники были сохранены в заявку")
        return "Бренд и модель техники были сохранены в заявку"

    async def save_circs_to_request(self, circs):
        self.logger.info(f"save_circs_to_request model: {circs}")
        await self.request_service.save_to_request(
            self.chat_id, f"Обстоятельства обращения: {circs}", "comment"
        )
        self.logger.info("Обстоятельства обращения были сохранены в заявку")
        return "Обстоятельства обращения были сохранены в заявку"

    def create_request(
        self, direction, latitude, longitude, address, phone, date, comment
    ):
        self.logger.info(
            f"create_request direction: {direction} latitude: {latitude} longitude: {longitude} address: {address} phone: {phone} date: {date} comment: {comment}"
        )
        token = os.environ.get("1С_TOKEN", "")

        with open("./data/template.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        params["order"]["client"]["display_name"] = "Владислав"
        params["order"]["uslugi_id"] = str(uuid4())

        params["order"]["services"][0]["service_id"] = direction
        params["order"]["client"]["phone"] = phone
        params["order"]["address"]["geopoint"]["latitude"] = latitude
        params["order"]["address"]["geopoint"]["longitude"] = longitude
        params["order"]["address"]["name"] = address
        params["order"]["desired_dt"] = date
        params["order"]["comment"] = comment
        self.logger.info(f"Parametrs: {params}")

        request_data = {"token": token, "params": params}
        url = f"{self.config['base_url']}/create_order"

        r = requests.post(
            url, json=request_data, headers={"Content-Type": "application/json"}
        )
        self.logger.info(f"Result:\n{r.status_code}\n{r.text}")

        if r.status_code == 200:
            return "Заявка была создана"
        else:
            return f"Ошибка при создании заявки: {r.text}"
