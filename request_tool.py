import requests
import logging
import json
import os
from uuid import uuid4
from langchain.tools.base import StructuredTool

from config_manager import ConfigManager
from file_service import FileService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

config_manager = ConfigManager("config.json")
request_service = FileService(config_manager.get("request_dir"), logger)


def create_request_tool(request):
    def create_request():
        base_url = config_manager.get("base_url")
        token = os.environ.get("1С_TOKEN", "")

        with open("./data/template.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        params["order"]["client"]["display_name"] = "Владислав"
        params["order"]["client"]["phone"] = request["phone"]
        params["order"]["address"]["geopoint"]["longitude"] = request["location"][
            "longitude"
        ]
        params["order"]["address"]["geopoint"]["latitude"] = request["location"][
            "latitude"
        ]
        params["order"]["address"]["name"] = request["address"]
        params["order"]["uslugi_id"] = str(uuid4())
        logger.info(f"Parametrs: {params}")

        request_data = {"token": token, "params": params}
        url = f"{base_url}/create_order"

        r = requests.post(url, json=request_data, headers={'Content-Type': 'application/json'})
        logger.info(f"Result:\n{r.status_code}\n{r.text}")

        if r.status_code == 200:
            return "Заявка была создана"
        else:
            return f"Ошибка при создании заявки: {r.text}"

    return StructuredTool.from_function(
        func=create_request,
        name="Создание заявки",
        description="Используйте, когда вам нужно создать заявку в 1С, используя полученную от пользователя информацию",
        return_direct=False,
    )