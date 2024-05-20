import logging
import json
import time
import pandas as pd
from requests.auth import HTTPBasicAuth
from requests import Session
from zeep import Client
from zeep.transports import Transport
import io


class OneC_Request:
    def __init__(self, login, password, clients):
        self.login = login
        self.password = password
        self.clients = clients
        self.session = Session()
        self.session.auth = HTTPBasicAuth(self.login, self.password)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def execute_query(self, query_params):
        results = {}

        for client_name, client_url in self.clients.items():
            start_time = time.time()
            client = Client(client_url, transport=Transport(session=self.session))
            self.logger.info(f"\n\nCalling ws for: {client_name}")

            response = client.service.ExecuteQuery(str(query_params))
            end_time = time.time()
            self.logger.info(f"Execution time: {end_time - start_time}")

            if "Ошибка" in str(response):
                self.logger.error(f"Error: {response}")
            else:
                data = json.loads(response)
                self.logger.info(f'Status OK: {data["Состояние"]}')
                if data["СтрОшибки"] != "":
                    self.logger.info(f'Error message: {data["СтрОшибки"]}')
                self.logger.info(f'Server-side duration: {data["Длительность"]/1000}')

                csv_table = data["ТаблицаЗначений"]
                id = pd.read_csv(io.StringIO(csv_table), sep=";")["id"].values
                results[client_name] = id

        self.logger.info("Done")
        return results
