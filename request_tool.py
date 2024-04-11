import requests
import logging
import json
from langchain.tools.base import StructuredTool

from config_manager import ConfigManager
from file_service import FileService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

config_manager = ConfigManager("config.json")
request_service = FileService(config_manager.get("request_dir"), logger)


import requests
import json
from langchain.tools.base import StructuredTool

def create_request_tool(request):
    def create_request():
        base_url = "https://service.icecorp.ru:7403"
        token = "1C-1fZ?ZXnPDKNccW5UwMZZHnq/Rj!Mv6TRWXaNSl6jFloC-drFMpLnnjOf/WPZ8-D77-15HKMta=qOiaE/wdEsMmEllOYW0Sma6YlJGM24GZNxciRdUOmwogTB8eeh!Q?dOXbQLT88-iC1eal2O6P0t7kZ-G7?BPkt4TECehAiNWRUgG7bA8S-O04o37?Yg3e6FyDf9fL2sXMMHIOU2RjQQpV43fnjqkOlfcjLdjojfZU1Spf=wY81YvHmeOHox"

        with open("template.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        params["order"]["client"]["display_name"] = "Андрей"
        params["order"]["client"]["phone"] = request["phone"]
        params["order"]["address"]["geopoint"]["longitude"] = request["address"]["longitude"]
        params["order"]["address"]["geopoint"]["latitude"] = request["address"]["latitude"]
        params["order"]["address"]["name"] = "Москва, Театральная площадь, д.1"
        params["order"]["uslugi_id"] = 2999
        logger.info(f"Parametrs: {params}")

        request_data = {
            'token': token,
            'params': params
        }

        url = f"{base_url}/create_order"

        r = requests.post(url, json=request_data, headers={'Content-Type': 'application/json'})
        """print('Result:')
        print(r.status_code)
        print(r.text)"""
        # Replace to logger
        logger.info(f"Result: {r.status_code}")
        logger.info(f"Result: {r.text}")


        if r.status_code == 200:
            return "Заявка была успешно создана"
        else:
            return f"Ошибка при создании заявки: {r.text}"

    return StructuredTool.from_function(
        func=create_request,
        name="Создание заявки",
        description="Используйте, когда вам нужно создать заявку в 1С, используя полученную от пользователя информацию",
        return_direct=False,
    )
