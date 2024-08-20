import os
import re
import time
import json
import requests
import numpy as np

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from datetime import datetime
from geopy.distance import geodesic
from geopy.geocoders import Nominatim, Yandex
from pydantic.v1 import BaseModel, Field
from telebot.types import ReplyKeyboardMarkup

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

from file_service import FileService
from config_manager import ConfigManager


# Definition args schemas for tools
class save_name_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    name: str = Field(description="name")


class save_direction_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    direction: str = Field(description="direction")


class save_circumstances_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    circumstances: str = Field(description="circumstances")


class save_brand_to_request_args(BaseModel):
    chat_id: int = Field(description="chat_id")
    brand: str = Field(description="brand")


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
    direction: str = Field(description="direction")
    circumstances: str = Field(description="circumstances")
    brand: str = Field(description="brand")
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
        oai_model,
        a_model,
        oai_temperature,
        a_temperature,
        chats_dir,
        request_dir,
        proxy_url,
        order_path,
        ws_paths,
        change_path,
        divisions,
        affilates,
        logger,
        bot_instance,
    ):
        self.logger = logger
        self.config = {
            "oai_model": oai_model,
            "a_model": a_model,
            "oai_temperature": oai_temperature,
            "a_temperature": a_temperature,
            "chats_dir": chats_dir,
            "request_dir": request_dir,
            "proxy_url": proxy_url,
            "order_path": order_path,
            "ws_paths": ws_paths,
            "change_path": change_path,
            "divisions": divisions,
            "affilates": affilates
        }
        self.agent_executor = None
        self.affilate = None
        self.bot_instance = bot_instance

        self.request_service = FileService(
            self.config["request_dir"],
            self.logger
        )
        self.chat_history_service = FileService(
            self.config["chats_dir"],
            self.logger
        )
        self.ban_manager = ConfigManager(
            "./data/banned_users.json",
            self.logger
        )

    def initialize_agent(self, company="OpenAI"):
        # Agent initialization depending on different LLMs
        self.company = company
        if self.company == "OpenAI":
            llm = ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=self.config["oai_model"],
                temperature=self.config["oai_temperature"]
            )
            self.logger.info(
                f'OpenAI ChatAgent init with model: {self.config["oai_model"]} and temperature: {self.config["oai_temperature"]}'
            )
        elif self.company == "Anthropic":
            llm = ChatAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                model=self.config["a_model"],
                temperature=self.config["a_temperature"]
            )
            self.logger.info(
                f'Anthropic ChatAgent init with model: {self.config["a_model"]} and temperature: {self.config["a_temperature"]}'
            )
        tools = []
        # Definition args schemas for tools

        # Tool: save_name_tool
        save_name_tool = StructuredTool.from_function(
            coroutine=self.save_name_to_request,
            name="Saving_name",
            description="""
                Сохраняет имя пользователя в новую заявку. Используйте этот инструмент ОБЯЗАТЕЛЬНО ВСЕГДА и СРАЗУ, если имеющееся у вас или полученное имя выглядит как настоящее человеческое.
Вам следует предоставить chat_id и непосредственно само name в качестве параметров
            """,
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
            description="""
                Сохраняет подходящее под запрос пользователя направление, причину обращения из имеющегося списка направлений в новую заявку. Нужно соотнести запрос и выбрать подходящее только из тех, что в этом списке.
Вам следует предоставить chat_id и непосредственно само direction из списка в качестве параметров
            """,
            args_schema=save_direction_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_direction_tool)

        # Tool: save_circumstances_tool
        save_circumstances_tool = StructuredTool.from_function(
            coroutine=self.save_circumstances_to_request,
            name="Saving_circumstances",
            description="""
                Сохраняет полученную дополнительную информацию по обстоятельствам, характеристикам обращения в новую заявку.
При возможном отказе в предоставлении подобной информации ничего сохранять и  вообще использовать этот инструмент не нужно!
Вам следует предоставить chat_id и непосредственно сами circumstances в качестве параметов
            """,
            args_schema=save_circumstances_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_circumstances_tool)

        # Tool: save_brand_tool
        save_brand_tool = StructuredTool.from_function(
            coroutine=self.save_brand_to_request,
            name="Saving_brand",
            description="""
                Сохраняет полученнный бренд / модель, ТОЛЬКО если речь идёт о технике, в новую заявку. Например, окна, двери или сантехника техникой НЕ являются, в таком случае использовать этот инструмент не нужно!
Вам следует предоставить chat_id и непосредственно сам brand из всего сообщения в качестве параметов
            """,
            args_schema=save_brand_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_brand_tool)

        # Tool: save_gps_tool
        save_gps_tool = StructuredTool.from_function(
            coroutine=self.save_gps_to_request,
            name="Saving_GPS-coordinates",
            description="""
                Сохраняет адрес на основании полученнных GPS-координат в новую заявку, только если ранее уже не был использован инструмент Saving_address.
Вам следует предоставить значения chat_id, latitude и longitude в качестве параметров
            """,
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
            description="""
                Сохраняет полученнный адрес в новую заявку. При сохранении убедитесь, что у вас есть ВСЕ три обязательных поля адреса (с городом, улицей, домом).
Вам следует предоставить chat_id и непосредственно сами значения в address из всего сообщения в качестве параметров, то есть без слов 'город', 'улица', 'дом' и т.д. Корпус и строение обозначайте одной буквой вместе с номером дома, например, 1к3 или 98с4, только так!
            """,
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
            description="""
                Сохраняет полученнную дополнительную информацию по адресу в новую заявку.
Вам следует предоставить chat_id и непосредственно сам address_line_2 из всего сообщения в качестве параметов
            """,
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
            description="""
                Сохраняет полученнный телефон в новую заявку.
Вам следует предоставить chat_id и непосредственно сам phone из всего сообщения в качестве параметов
            """,
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
            description="""
                Сохраняет нужную дату визита в новую заявку. Вам следует САМИМ предоставить в инструмент chat_id и непосредственно саму date в формате 'yyyy-mm-ddT00:00Z', определённую вами самостоятельно по умолчанию или же полученную из сообщения пользователя в качестве параметров.
ПРИНИМАЙТЕ дату от пользователя в ЛЮБОМ свободном формате (например, 'сегодня' или 'завтра'), а НЕ в том, что выше. Главное используйте сами потом в указанном, отформатировав при необходимости
            """,
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
            description="""
                Сохраняет любые полезные по вашему мнению комментарии пользователя, а также важную информацию (например, факт того, что была озвучена стоимость диагностики) в диалоге в новую заявку. Ни в коем случае НЕЛЬЗЯ передавать здесь информацию, содержающую детали адреса (квартира, подъезд и т.п.) или ЛЮБЫЕ телефоны клиента, даже если он просит, в таком случае НЕ используйте этот инструмент.
Вам следует предоставить chat_id и comment своими словами в качестве параметров
            """,
            args_schema=save_comment_to_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(save_comment_tool)

        # Tool: create_request_tool
        create_request_tool = StructuredTool.from_function(
            coroutine=self.create_request,
            name="Create_request",
            description="""
                Создает полностью заполненную новую заявку в 1С и при доступности определяет её номер.
Вам следует предоставить chat_id и по отдельности сами значения ключей словаря (request) с текущей заявкой из вашего системного промпта в качестве соответствующих параметров инструмента, кроме ключа address_line_2. Из его же значения выделите и передайте отдельно при наличии непосредственно сами численно-буквенные значения apartment, entrance, floor и intercom (т.е. без слов) из всего address_line_2 в качестве остальных соответствующих параметров инструмента.
Из address же передавайте непосредственно сами значения в качестве параметров, то есть без слов 'город', 'улица', 'дом' и т.д. Корпус и строение обозначайте одной буквой вместе с номером дома, например, 1к3 или 98с4, только так!
При отсутствии у вас значений каких-либо параметров передавайте на их месте просто пустые строки - ''
            """,
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
            description="""
                Находит и предоставляет пользователю список его ОФОРМЛЕННЫХ заявок для выбора, чтобы определить контекст всего диалога, если речь идёт уже о каких-либо созданных заявках, а НЕ об оформлении новой. Используйте этот инструмент ВСЕГДА ОБЯЗАТЕЛЬНО, когда спрашивайте номер заявки у пользователя, но ТОЛЬКО ОДИН РАЗ, когда вам нужно понять, о какой именно заявке идёт речь, например, СРАЗУ, как только пользователь захочет изменить или дополнить данные по уже существующей заявке.
Если вы уже явно получили от пользователя номер заявки, повторно НИ В КОЕМ СЛУЧАЕ НЕ используйте этот инструмент! Вам следует предоставить chat_id в качестве параметра
            """,
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
            description="""
                Изменяет нужные данные / значения полей в уже СУЩЕСТВУЮЩЕЙ заявке. Допустимо обрабатывать ТОЛЬКО ТЕЛЕФОН или ЛЮБУЮ ДОПОЛНИТЕЛЬНУЮ ИНФОРМАЦИЮ КАК КОММЕНТАРИЙ. Для редактирования уже имеющихся СОЗДАННЫХ заявок используйте ТОЛЬКО ЭТОТ инструмент, а НЕ обычные с добавлением информации в новую!
Вам следует предоставить сам номер текущей заявки request_number; field_name - подходящее название поля: 'comment' или 'phone'; а также само новое значение поля, полученное от пользователя (field_value) в качестве параметров
            """,
            args_schema=change_request_args,
            return_direct=False,
            handle_tool_error=True,
            handle_validation_error=True,
            verbose=True,
        )
        tools.append(change_request_tool)

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
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            max_iterations=20,
            return_intermediate_steps=True
        )
    
    def distance_calculation(self, latitude, longitude, affilate_coordinates):
        distance = np.inf
        affilate = None
        for aff, boundaries in affilate_coordinates:
            for coordinates in boundaries:
                if geodesic(
                    [latitude, longitude],
                    coordinates
                ).kilometers < distance:
                    distance = geodesic(
                        [latitude, longitude],
                        coordinates
                    ).kilometers
                    affilate = aff
        return distance, affilate

    async def save_name_to_request(self, chat_id, name):
        self.logger.info(f"save_name_to_request name: {name}")
        try:
            await self.request_service.save_to_request(chat_id, name, "name")
        except Exception as e:
            self.logger.error(f"Error in saving customer name: {e}")
        self.logger.info("Customer name was saved in the request")
        return "Имя пользователя было сохранено в заявку"

    async def save_direction_to_request(self, chat_id, direction):
        self.logger.info(f"save_direction_to_request direction: {direction}")
        if direction not in self.config["divisions"].values():
            return "Выбрано некорректное направление обращения, определите сами повторно подходящее именно из вашего списка"
        try:
            await self.request_service.save_to_request(
                chat_id,
                direction,
                "direction"
            )
        except Exception as e:
            self.logger.error(f"Error in saving direction: {e}")
            return f"Ошибка при сохранении направления обращения: {e}"
        self.logger.info("Direction was saved in the request")
        return "Направление, причина обращения было сохранено в заявку"

    async def save_circumstances_to_request(self, chat_id, circumstances):
        self.logger.info(f"save_circumstances_to_request circumstances: {circumstances}")
        try:
            await self.request_service.save_to_request(
                chat_id,
                circumstances,
                "circumstances"
            )
        except Exception as e:
            self.logger.error(f"Error in saving circumstances: {e}")
        self.logger.info("Circumstances was saved in the request")
        return "Обстоятельства обращения были сохранены в заявку"

    async def save_brand_to_request(self, chat_id, brand):
        self.logger.info(f"save_brand_to_request brand: {brand}")
        try:
            await self.request_service.save_to_request(
                chat_id,
                brand,
                "brand"
            )
        except Exception as e:
            self.logger.error(f"Error in saving brand: {e}")
        self.logger.info("Brand was saved in the request")
        return "Бренд / модель были сохранены в заявку"

    async def save_gps_to_request(self, chat_id, latitude, longitude):
        self.logger.info(
            f"save_gps_to_request latitude: {latitude} longitude: {longitude}"
        )
        try:
            distance, self.affilate = self.distance_calculation(
                latitude,
                longitude,
                self.config["affilates"].items()
            )
            if (
                self.affilate == "Москва" and distance > 100
            ) or (
                self.affilate != "Москва" and distance > 90
            ):
                return "Указанный пользователем адрес находится вне зоны работы компании. Вежливо донесите это до пользователя и прекратите далее оформлять заявку!"
            elif (
                self.affilate == "Москва" and distance > 50
            ) or (
                self.affilate != "Москва" and distance > 40
            ):
                return "Указанный пользователем адрес находится вне зоны бесплатного выезда мастера. Предложите пользователю связаться с нами по нашему контактному телефону 8 495 723 723 0, указав его, и прекратите далее оформлять заявку!"
        except Exception as e:
            self.logger.error(f"Error in distance calculation: {e}")

        try:
            try:
                geolocator = Nominatim(user_agent="my_app")
                address = geolocator.reverse(f"{latitude}, {longitude}").address
            except Exception as e:
                self.logger.error(
                    f"Error in getting address: {e}, using Yandex geolocator"
                )
                geolocator = Yandex(
                    api_key=os.environ.get("YANDEX_GEOCODER_KEY", "")
                )
                address = geolocator.reverse(
                    f"{latitude}, {longitude}"
                ).address

            await self.request_service.save_to_request(
                chat_id,
                latitude,
                "latitude"
            )
            await self.request_service.save_to_request(
                chat_id,
                longitude,
                "longitude"
            )
            await self.request_service.save_to_request(
                chat_id,
                address,
                "address"
            )
        except Exception as e:
            self.logger.error(f"Error in saving address: {e}")
            return f"Ошибка при сохранении адреса: {e}"
        
        self.logger.info("Address was saved in the request")
        return "Адрес пользователя был сохранен в заявку"

    async def save_address_to_request(self, chat_id, address):
        del_pattern = re.compile(
            r"улица\s*|ул\.*\s|дом(\s|,)|д\.*\s|город(\s|,)|гор\.*\s|г\.*\s",
            re.IGNORECASE
        )
        ch_pattern = r'(,*\sстроение|,*\sстр\.*|,*\sс\.*)\s(\d+)|(,*\sкорпус|,*\sкорп\.*|,*\sк\.*)\s(\d+)'
        replacement = lambda m: f"с{m.group(2)}" if m.group(1) else f"к{m.group(4)}" if m.group(3) else m.group(0)

        address = re.sub(ch_pattern, replacement, address, flags=re.IGNORECASE)
        address = re.sub(del_pattern, '', address)

        self.logger.info(f"save_address_to_request address: {address}")
        try:
            try:
                geolocator = Nominatim(user_agent="my_app")
                location = geolocator.geocode(address)
                latitude = location.latitude
                longitude = location.longitude
            except Exception as e:
                self.logger.error(
                    f"Error in geocoding address: {e}, using Yandex geolocator"
                )
                geolocator = Yandex(
                    api_key=os.environ.get("YANDEX_GEOCODER_KEY", "")
                )
                location = geolocator.geocode(address)
                latitude = location.latitude
                longitude = location.longitude
        except Exception as e:
            self.logger.error(
                f"Error in geocoding address: {e}")
            return "Не удалось получить координаты адреса. Запросите адрес ещё раз"
        
        try:
            distance, self.affilate = self.distance_calculation(
                latitude,
                longitude,
                self.config["affilates"].items()
            )
            if (
                self.affilate == "Москва" and distance > 100
            ) or (
                self.affilate != "Москва" and distance > 90
            ):
                return "Указанный пользователем адрес находится вне зоны работы компании. Вежливо донесите это до пользователя и прекратите далее оформлять заявку!"
            elif (
                self.affilate == "Москва" and distance > 50
            ) or (
                self.affilate != "Москва" and distance > 40
            ):
                return "Указанный пользователем адрес находится вне зоны бесплатного выезда мастера. Предложите пользователю связаться с нами по нашему контактному телефону 8 495 723 723 0, указав его, и прекратите далее оформлять заявку!"
        except Exception as e:
            self.logger.error(f"Error in distance calculation: {e}")
        
        try:
            try:
                geolocator = Nominatim(user_agent="my_app")
                address = geolocator.reverse(f"{latitude}, {longitude}").address
            except Exception as e:
                self.logger.error(
                    f"Error in getting address: {e}, using Yandex geolocator"
                )
                geolocator = Yandex(
                    api_key=os.environ.get("YANDEX_GEOCODER_KEY", "")
                )
                address = geolocator.reverse(
                    f"{latitude}, {longitude}"
                ).address

            await self.request_service.save_to_request(
                chat_id,
                latitude,
                "latitude"
            )
            await self.request_service.save_to_request(
                chat_id,
                longitude,
                "longitude"
            )
            await self.request_service.save_to_request(
                chat_id,
                address,
                "address"
            )
        except Exception as e:
            self.logger.error(f"Error in saving address: {e}")
            return f"Ошибка при сохранении адреса: {e}"
        
        self.logger.info("Address was saved in the request")
        return "Адрес пользователя был сохранен в заявку"

    async def save_address_line_2_to_request(self, chat_id, address_line_2):
        self.logger.info(
            f"save_address_line_2_to_request address: {address_line_2}"
        )
        try:
            await self.request_service.save_to_request(
                chat_id,
                address_line_2,
                "address_line_2"
            )
        except Exception as e:
            self.logger.error(f"Error in saving address line 2: {e}")
        self.logger.info("Address line 2 was saved in the request")
        return "Вторая линия адреса пользователя была сохранена в заявку"

    async def save_phone_to_request(self, chat_id, phone):
        self.logger.info(f"save_phone_to_request phone: {phone}")
        phone = re.sub(r"[^\d]", "", phone)
        if len(phone) < 10:
            return "Пользователь предоставил некорректный номер телефона, запросите его ещё раз. Передайте, что возможно, не хватает кода оператора / города"
        
        try:
            await self.request_service.save_to_request(
                chat_id,
                "".join(re.findall(r"[\d]", phone)),
                "phone"
            )
        except Exception as e:
            self.logger.error(f"Error in saving phone: {e}")
            return f"Ошибка при сохранении телефона: {e}"
        self.logger.info("Phone was saved in the request")
        return "Телефон пользователя был сохранен в заявку"

    async def save_date_to_request(self, chat_id, date):
        self.logger.info(f"save_date_to_request date: {date}")
        try:
            await self.request_service.save_to_request(chat_id, date, "date")
        except Exception as e:
            self.logger.error(f"Error in saving date: {e}")
            return f"Ошибка при сохранении даты посещения: {e}"
        self.logger.info("Date was saved in the request")
        return "Дата посещения была сохранена в заявку"

    async def save_comment_to_request(self, chat_id, comment):
        self.logger.info(f"save_comment_to_request comment: {comment}")
        try:
            await self.request_service.save_to_request(
                chat_id,
                comment,
                "comment"
            )
        except Exception as e:
            self.logger.error(f"Error in saving comment: {e}")
        self.logger.info("Comment was saved in the request")
        return "Комментарий был сохранен в заявку"

    async def create_request(
        self,
        chat_id,
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
        comment="",
        circumstances="",
        brand=""
    ):
        token = os.environ.get("1С_TOKEN", "")
        login = os.environ.get("1C_LOGIN", "")
        password = os.environ.get("1C_PASSWORD", "")

        try:
            if await self.request_selection(
                chat_id,
                request_creating=True
            ) == "Ban":
                self.ban_manager.set(
                    chat_id,
                    time.strftime("%Y-%m-%d %H:%M", time.localtime())
                )
                return """Пользователем создано подозрительное число заявок за день.
                Передайте ему это, а также то, что в целях безопасности ему необходимо оформлять далее заявки с другого Телеграм аккаунта. И прекратите далее оформлять заявку!
                """
        except Exception as error:
            self.logger.error(
                f"Error in receiving today customer requests: {error}"
            )

        try:
            with open("./data/template.json", "r", encoding="utf-8") as f:
                order_params = json.load(f)
        except Exception as e:
            self.logger.error(f"Error in getting params template: {e}")
            return f"Ошибка при получении шаблона параметров заявки: {e}"
        
        try:
            order_params["order"]["client"]["display_name"] = name
            order_params["order"]["address"]["floor"] = floor
            order_params["order"]["address"]["entrance"] = entrance
            order_params["order"]["address"]["apartment"] = apartment
            order_params["order"]["address"]["intercom"] = intercom
        except Exception as e:
            self.logger.error(f"Error in getting order params: {e}")
        
        if latitude == 0 and longitude == 0:
            try:
                latitude = (await self.request_service.read_request(chat_id))["latitude"]
                longitude = (await self.request_service.read_request(chat_id))["longitude"]
            except Exception as e:
                self.logger.error(
                    f"Error in reading current request files: {e}"
                )
                return "Вы не сохранили адрес! Перед 'Create_request' используйте сначала остальные инструменты для сохранения всех полученных данных"
        if direction not in self.config["divisions"].values():
            return "Выбрано некорректное направление обращения, определите сами повторно подходящее именно из вашего списка"
        
        for detail in [brand, circumstances]:
            if detail !="":
                comment += f"\n{detail}"

        # Double-check of personal data
        try:
            if self.company == "OpenAI":
                client = AsyncOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY", "")
                )
                response = await client.chat.completions.create(
                    model=self.config["oai_model"],
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
            elif self.company == "Anthropic":
                client = AsyncAnthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                )
                response = await client.messages.create(
                    model=self.config["a_model"],
                    temperature=0,
                    system="Вы - сотрудник по сохранности конфиденциальных данных. В передаваемом вами тексте никогда не должно быть никакой следующей информации: любых номеров телефонов; значений подъезда, этажа, квартиры, домофона. Возвращайте в ответе ТОЛЬКО полученный текст с УБРАННОЙ всей перечисленной выше информацией, НИ В КОЕМ СЛУЧАЕ НЕ ваш ответ с размышлениями. Если текст изначально пустой, также возвращайте пустую строку - ''.",
                    messages=[
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
                comment = response.content.text
        except Exception as e:
            self.logger.error(f"Error in checking personal data: {e}")

        pattern = re.compile(
            r"([+]?[\d]?\d{3}.*?\d{3}.*?\d{2}.*?\d{2})|подъезд|этаж|эт|квартир|кв|домофон|код",
            re.IGNORECASE
        )
        comment = re.sub(pattern, '', comment)

        if not self.affilate:
            try:
                distance, self.affilate = self.distance_calculation(
                    latitude,
                    longitude,
                    self.config["affilates"].items()
                )
                if (
                    self.affilate == "Москва" and distance > 100
                ) or (
                    self.affilate != "Москва" and distance > 90
                ):
                    return """Указанный пользователем адрес находится вне зоны работы компании.
                    Вежливо донесите это до пользователя и прекратите далее оформлять заявку!"""
                elif (
                    self.affilate == "Москва" and distance > 50
                ) or (
                    self.affilate != "Москва" and distance > 40
                ):
                    return """Указанный пользователем адрес находится вне зоны бесплатного выезда мастера.
                    Предложите пользователю связаться с нами по нашему контактному телефону 8 495 723 723 0, указав его, и прекратите далее оформлять заявку!"""
            except Exception as e:
                self.logger.error(f"Error in distance calculation: {e}")

        del_pattern = re.compile(
            r"улица\s*|ул\.*\s|дом(\s|,)|д\.*\s|город(\s|,)|гор\.*\s|г\.*\s",
            re.IGNORECASE
        )
        ch_pattern = r'(,*\sстроение|,*\sстр\.*|,*\sс\.*)\s(\d+)|(,*\sкорпус|,*\sкорп\.*|,*\sк\.*)\s(\d+)'
        replacement = lambda m: f"с{m.group(2)}" if m.group(1) else f"к{m.group(4)}" if m.group(3) else m.group(0)

        address = re.sub(ch_pattern, replacement, address, flags=re.IGNORECASE)
        address = re.sub(del_pattern, '', address)

        try:
            order_params["order"]["services"][0]["service_id"] = direction
            order_params["order"]["desired_dt"] = date
            order_params["order"]["client"]["phone"] = phone
            order_params["order"]["address"]["name"] = address
            order_params["order"]["address"]["name_components"][0]["name"] = self.affilate
            order_params["order"]["comment"] = comment
            order_params["order"]["address"]["geopoint"]["latitude"] = latitude
            order_params["order"]["address"]["geopoint"]["longitude"] = longitude
            order_params["order"]["uslugi_id"] = time.strftime(
                "%Y-%m-%d-%H-%M-%S",
                time.localtime()
            ).replace("-", "")+str(chat_id)
        except Exception as e:
            self.logger.error(f"Error in getting order params: {e}")
            return f"Ошибка при получении параметров заявки: {e}"

        try:
            ws_params = {
                "Идентификатор": "new_bid_number",
                "НомерПартнера": order_params["order"]["uslugi_id"],
            }
            order_url = f"{self.config['proxy_url']}/hs"
            ws_url = f"{self.config['proxy_url']}/ws"

            order_data = {
                "clientPath": self.config["order_path"]
            }
            ws_data = {
                "clientPath": self.config["ws_paths"],
                "login": login,
                "password": password,
            }
            request_number = None
        except Exception as e:
            self.logger.error(f"Error in getting web services params: {e}")
            return f"Ошибка при получении параметров вэб-сервисов: {e}"
        
        try:
            order = requests.post(
                order_url,
                json={
                    "config": order_data,
                    "params": order_params,
                    "token": token
                }
            )
        except Exception as e:
            self.logger.error(f"Error in creating request: {e}")
            return f"Ошибка при создании заявки: {e}"
        finally:
            if order:
                self.logger.info(f"Result:\n{order.status_code}\n{order.text}")

        try:
            # Receiving number of new request
            results = requests.post(
                ws_url,
                json={"config": ws_data, "params": ws_params, "token": token}
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
            self.affilate = None
            if request_number:
                return f"Заявка была создана с номером {request_number}"
            else:
                return "Заявка была создана"
        else:
            self.logger.error(f"Error in creating request: {order.text}")
            return f"Ошибка при создании заявки: {order.text}"

    async def request_selection(self, chat_id, request_creating=False):
        token = os.environ.get("1С_TOKEN", "")
        login = os.environ.get("1C_LOGIN", "")
        password = os.environ.get("1C_PASSWORD", "")
        request_creating=request_creating

        try:
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
            request_numbers = {}
            divisions = self.config["divisions"]
        except Exception as e:
            self.logger.error(f"Error in getting web service params: {e}")
            return f"Ошибка при получении параметров вэб-сервиса: {e}"

        try:
            results = requests.post(
                ws_url,
                json={"config": ws_data, "params": ws_params, "token": token}
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
            self.logger.error(f"Error in receiving request numbers: {e}")
            return f"Ошибка при получении списка заявок: {e}"

        # Cheking the count of request in today
        if len(request_numbers) >= 3 and request_creating==True:
            if sum(1 for request in request_numbers.values() if request['date'] == time.strftime("%d.%m.%Y", time.localtime())) >= 3:
                return "Ban"
        
        if request_creating==False:
            if len(request_numbers) > 0:
                markup = ReplyKeyboardMarkup(
                    resize_keyboard=True,
                    one_time_keyboard=True
                )
                text = "Секунду..."
                for number, values in request_numbers.items():
                    markup.add(
                        f"Заявка {number} от {values['date']}; {values['division']}"
                    )
                markup.add("🏠 Вернуться в меню")
                await self.bot_instance.send_message(
                    chat_id,
                    text,
                    reply_markup=markup
                )
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
        locality = None

        try:
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
        except Exception as e:
            self.logger.error(f"Error in getting web service params: {e}")
            return f"Ошибка при получении параметров вэб-сервиса: {e}"
        
        # Unloading items critical for change
        try:
            results = requests.post(
                ws_url,
                json={"config": ws_data, "params": ws_params, "token": token}
            ).json()["result"]
            self.logger.info(f"results: {results}")
            
            for value in results.values():
                if len(value) > 0:
                    partner_number = str(value[0]["id"])
                    date_received = value[0]["date"]
                    date = datetime.strptime(
                        date_received,
                        '%d.%m.%Y %H:%M:%S'
                    )
                    date_str = date.strftime('%Y-%m-%dT%H:%MZ')
                    if value[0]["comment"]:
                        if value[0]["comment"] == "''":
                            comment = ''
                        else:
                            comment = value[0]["comment"]
                    else:
                        comment = ''
                    break

            get_url = f"{self.config['proxy_url']}/rev"
            get_data = {
                "clientPath": {"crm": self.config["order_path"]["crm"]+partner_number}
            }
            request = requests.post(
                get_url,
                json={"config": get_data, "token": token}
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
            try:
                with open("./data/template.json", "r", encoding="utf-8") as f:
                    change_params = json.load(f)
            except Exception as e:
                self.logger.error(f"Error in getting params template: {e}")
                return f"Ошибка при получении шаблона параметров заявки: {e}"
            
            change_params["order"]["uslugi_id"] = partner_number
            change_params["order"]["desired_dt"] = date_str
            change_params["order"]["address"]["name_components"][0]["name"] = locality
            change_params["order"]["revision"] = revision + 1

            # Validation of value types
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
            
            # Change of request
            try:
                change_url = f"{self.config['proxy_url']}/ex"
                change_data = {
                    "clientPath": {"crm": self.config["change_path"]["crm"]+partner_number}
                }
                change = requests.post(
                    change_url,
                    json={
                        "config": change_data,
                        "params": change_params,
                        "token": token
                    }
                )
            except Exception as e:
                self.logger.error(f"Error in changing request: {e}")
                return f"Ошибка при обновлении заявки: {e}"
            self.logger.info(f"Result:\n{change.status_code}\n{change.text}")
            
            if change.status_code == 200:
                return f"Данные заявки были обновлены"
            else:
                self.logger.error(f"Error in changing request: {change.text}")
                return f"Ошибка при обновлении заявки: {change.text}"
        else:
            return f"Произошла ошибка при получении данных заявки: {e}"