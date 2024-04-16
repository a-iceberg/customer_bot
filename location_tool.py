from langchain.tools.base import StructuredTool
from file_service import save_to_request
from geopy.geocoders import Nominatim


def create_location_tool(chat_id, message):
    def save_location():
        if "location" in message:
            geolocator = Nominatim(user_agent="my_app")
            location = message["location"]
            address = geolocator.reverse(
                f'{location["latitude"]}, {location["longitude"]}'
            ).address
            save_to_request(
                chat_id, message["location"], message["message_id"], "location"
            )
            save_to_request(chat_id, address, message["message_id"], "address")
        else:
            save_to_request(chat_id, message["text"], message["message_id"], "address")
        return "Адрес пользователя был получен"

    return StructuredTool.from_function(
        func=save_location,
        name="Сохранение адреса",
        description="Используйте, когда вам нужно сохранить полученный от пользователя адрес в заявку",
        return_direct=False,
    )
