import json
import os
import shutil
from pathlib import Path
import time as py_time
import aiofiles
from langchain.schema import AIMessage, HumanMessage


class FileService:
    def __init__(self, data_dir, logger):
        self.data_dir = data_dir
        self.logger = logger

    def file_path(self, chat_id):
        return os.path.join(self.data_dir, str(chat_id))

    async def save_to_chat_history(
        self,
        chat_id,
        message_text,
        message_id,
        message_type,
        user_name,
        event_id="default",
        date_override=None,
    ):
        self.logger.info(
            f"[{event_id}] Saving message to chat history for chat_id: {chat_id} for message_id: {message_id}"
        )
        if date_override is None:
            message_date = py_time.strftime("%Y-%m-%d-%H-%M-%S", py_time.localtime())
            parsed_time = py_time.strptime(message_date, "%Y-%m-%d-%H-%M-%S")
            unix_timestamp = int(py_time.mktime(parsed_time))
        else:
            unix_timestamp = date_override
            message_date = py_time.strftime(
                "%Y-%m-%d-%H-%M-%S", py_time.localtime(unix_timestamp)
            )
        log_file_name = f"{unix_timestamp}_{message_id}_{event_id}.json"

        chat_log_dir = self.file_path(chat_id)
        Path(chat_log_dir).mkdir(parents=True, exist_ok=True)

        full_path = os.path.join(chat_log_dir, log_file_name)
        async with aiofiles.open(full_path, "w") as log_file:
            await log_file.write(
                json.dumps(
                    {
                        "type": message_type,
                        "text": message_text,
                        "date": message_date,
                        "message_id": message_id,
                        "name_of_user": user_name,
                    },
                    ensure_ascii=False,
                )
            )

    async def read_chat_history(self, chat_id: str):
        """Reads the chat history from a folder and returns it as a list of messages."""
        chat_history = []
        chat_log_path = self.file_path(chat_id)
        Path(chat_log_path).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Reading chat history from: {chat_log_path}")

        for log_file in sorted(os.listdir(chat_log_path)):
            full_path = os.path.join(chat_log_path, log_file)
            try:
                with open(full_path, "r") as file:
                    message = json.load(file)
                    if message["type"] == "AIMessage":
                        chat_history.append(AIMessage(content=message["text"]))
                    elif message["type"] == "HumanMessage":
                        chat_history.append(HumanMessage(content=message["text"]))
            except Exception as e:
                self.logger.error(f"Error reading chat history file {log_file}: {e}")
                # Remove problematic file
                os.remove(full_path)

        return chat_history

    def delete_chat_history(self, chat_id: str):
        """Deletes the chat history folder and all its content."""
        chat_log_path = Path(self.file_path(chat_id))
        if chat_log_path.exists() and chat_log_path.is_dir():
            try:
                shutil.rmtree(chat_log_path)
                self.logger.info(f"Deleted chat history for chat_id: {chat_id}")
            except Exception as e:
                self.logger.error(
                    f"Error deleting chat history for chat_id: {chat_id}: {e}"
                )
        else:
            self.logger.info(
                f"No chat history found for chat_id: {chat_id}, nothing to delete."
            )

    async def save_to_request(
        self,
        chat_id,
        message_text,
        message_type,
        date_override=None,
    ):
        self.logger.info(
            f"[{message_type}] Saving request item to request for chat_id: {chat_id}"
        )
        if date_override is None:
            message_date = py_time.strftime("%Y-%m-%d-%H-%M-%S", py_time.localtime())
            parsed_time = py_time.strptime(message_date, "%Y-%m-%d-%H-%M-%S")
            unix_timestamp = int(py_time.mktime(parsed_time))
        else:
            unix_timestamp = date_override
            message_date = py_time.strftime(
                "%Y-%m-%d-%H-%M-%S", py_time.localtime(unix_timestamp)
            )
        log_file_name = f"{unix_timestamp}_{message_type}.json"

        request_dir = self.file_path(chat_id)
        Path(request_dir).mkdir(parents=True, exist_ok=True)

        full_path = os.path.join(request_dir, log_file_name)
        async with aiofiles.open(full_path, "w") as log_file:
            await log_file.write(
                json.dumps(
                    {
                        "type": message_type,
                        "text": message_text,
                        "date": message_date,
                    },
                    ensure_ascii=False,
                )
            )

    async def read_request(self, chat_id: str):
        """Reads request items from a folder and returns it."""
        request_items = {}
        request_path = self.file_path(chat_id)
        Path(request_path).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Reading request from: {request_path}")

        for item in sorted(os.listdir(request_path)):
            full_path = os.path.join(request_path, item)
            try:
                with open(full_path, "r") as file:
                    message = json.load(file)
                    if message["type"] == "category":
                        request_items["category"] = message["text"]
                    elif message["type"] == "phone":
                        request_items["phone"] = message["text"]
                    elif message["type"] == "location":
                        request_items["location"] = message["text"]
                    elif message["type"] == "address":
                        request_items["address"] = message["text"]
            except Exception as e:
                self.logger.error(f"Error reading request file {item}: {e}")
                # Remove problematic file
                os.remove(full_path)

        return request_items
