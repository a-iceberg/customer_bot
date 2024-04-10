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


def create_request_tool(request):
    def create_request():
        url = config_manager.get("1С_url")
        with open("template.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        params["order"]["client"]["display_name"] = "Андрей"
        params["order"]["client"]["phone"] = request["phone"]
        params["order"]["address"]["geopoint"]["longitude"] = request["address"][
            "longitude"
        ]
        params["order"]["address"]["geopoint"]["latitude"] = request["address"][
            "latitude"
        ]
        params["order"]["address"]["name"] = "Москва, Театральная площадь, д.1"
        params["order"]["uslugi_id"] = 2999
        logger.info(f"Parametrs: {params}")

        r = requests.post(
            url, json=params, headers={"Content-Type": "application/json"}
        )
        logger.info(f"Result:\n{r.status_code}\n{r.text}")
        return "Заявка была создана"

    return StructuredTool.from_function(
        func=create_request,
        name="Создание заявки",
        description="Используйте, когда вам нужно создать заявку в 1С, используя полученную от пользователя информацию",
        return_direct=False,
    )
