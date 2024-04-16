import logging
from langchain.tools.base import StructuredTool
from config_manager import ConfigManager
from file_service import FileService
from geopy.geocoders import Nominatim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

config_manager = ConfigManager("config.json")
request_service = FileService(config_manager.get("request_dir"), logger)


def create_location_tool(chat_id, message):

    async def save_location():
        if "location" in message:
            geolocator = Nominatim(user_agent="my_app")
            location = message["location"]
            address = geolocator.reverse(
                f'{location["latitude"]}, {location["longitude"]}'
            ).address
            await request_service.save_to_request(
                chat_id, message["location"], message["message_id"], "location"
            )
            await request_service.save_to_request(
                chat_id, address, message["message_id"], "address"
            )
        else:
            await request_service.save_to_request(
                chat_id, message["text"], message["message_id"], "address"
            )
        return "Адрес пользователя был получен"

    return StructuredTool.from_function(
        func=save_location,
        name="Сохранение адреса",
        description="Используйте, когда вам нужно сохранить полученный от пользователя адрес в заявку",
        return_direct=False,
    )
