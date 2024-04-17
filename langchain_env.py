from langchain.tools.base import StructuredTool
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
import os
import re
from langchain.agents import initialize_agent, AgentType
from geopy.geocoders import Nominatim
import requests
import json
from uuid import uuid4

from file_service import FileService


class save_gps_to_request_args(BaseModel):
    latitude: str = Field(description="latitude")
    longitude: str = Field(description="longitude")


class save_address_to_request_args(BaseModel):
    address: str = Field(description="address")


class save_phone_to_request_args(BaseModel):
    phone: str = Field(description="phone")


class create_request_args(BaseModel):
    request: dict = Field(description="request")


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
        llm = ChatAnthropic(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=self.config["model"],
            temperature=self.config["temperature"],
        )
        tools = []

        # Tool: save_gps_tool
        save_gps_tool = StructuredTool.from_function(
            func=self.save_gps_to_request,
            name="Сохранение GPS-координат",
            description="Сохраняет адрес на основании полученнных GPS-координат в заявку. Вам следует предоставить latitude и longitude в качестве параметров.",
            args_schema=save_gps_to_request_args,
            return_direct=False,
        )
        tools.append(save_gps_tool)

        # Tool: save_address_tool
        save_address_tool = StructuredTool.from_function(
            func=self.save_address_to_request,
            name="Сохранение адреса",
            description="Сохраняет полученнный адрес в заявку. Вам следует предоставить только непосредственно сам address из всего сообщения в качестве параметра.",
            args_schema=save_address_to_request_args,
            return_direct=False,
        )
        tools.append(save_address_tool)

        # Tool: save_phone_tool
        save_phone_tool = StructuredTool.from_function(
            func=self.save_phone_to_request,
            name="Сохранение телефона",
            description="Сохраняет полученнный телефон в заявку. Вам следует предоставить только непосредственно сам phone из всего сообщения в качестве параметра.",
            args_schema=save_phone_to_request_args,
            return_direct=False,
        )
        tools.append(save_phone_tool)

        # Tool: request_tool
        request_tool = StructuredTool.from_function(
            func=self.create_request,
            name="Создание заявки",
            description="Создает полностью заполненную заявку в 1С. Вам следует предоставить request в качестве параметра.",
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

    async def save_gps_to_request(self, latitude, longitude):
        self.logger.info(
            f"save_gps_to_request latitude: {latitude} longitude: {longitude}"
        )
        geolocator = Nominatim(user_agent="my_app")
        address = geolocator.reverse(f"{latitude}, {longitude}").address
        await self.request_service.save_to_request(self.chat_id, address, "address")
        self.logger.info("Адрес пользователя был сохранен в заявку")
        return "Адрес пользователя был сохранен в заявку"

    async def save_address_to_request(self, address):
        self.logger.info(f"save_address_to_request address: {address}")
        await self.request_service.save_to_request(self.chat_id, address, "address")
        self.logger.info("Адрес пользователя был сохранен в заявку")
        return "Адрес пользователя был сохранен в заявку"

    async def save_phone_to_request(self, phone):
        self.logger.info(f"save_phone_to_request phone: {phone}")
        await self.request_service.save_to_request(
            self.chat_id, "".join(re.findall(r"[+\d]", phone)), "phone"
        )
        self.logger.info("Телефон пользователя был сохранен в заявку")
        return "Телефон пользователя был сохранен в заявку"

    def create_request(self, request):
        self.logger.info(f"create_request request: {request}")
        token = os.environ.get("1С_TOKEN", "")

        with open("./data/template.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        params["order"]["client"]["display_name"] = "Владислав"
        params["order"]["client"]["phone"] = request["phone"]
        params["order"]["address"]["geopoint"]["longitude"] = 0
        params["order"]["address"]["geopoint"]["latitude"] = 0
        params["order"]["address"]["name"] = request["address"]
        params["order"]["uslugi_id"] = str(uuid4())
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
