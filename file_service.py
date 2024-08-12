import os
import time
import json
import shutil
import aiofiles

from pathlib import Path
from pyrogram import Client
from datetime import datetime
from psycopg_pool import AsyncConnectionPool
from langchain.schema import AIMessage, HumanMessage


class FileService:
    def __init__(self, data_dir, logger):
        self.data_dir = data_dir
        self.logger = logger
        self.chat_history_client = None
        self.pool = None

    def file_path(self, chat_id):
        return os.path.join(self.data_dir, str(chat_id))

    async def save_message_id(self, chat_id, message_id):
        chat_dir = self.file_path(chat_id)
        Path(chat_dir).mkdir(parents=True, exist_ok=True)
        full_path = os.path.join(chat_dir, 'chat_data.json')

        data = {
            "message_id": message_id,
            "chat_history_date": time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime()
            )
        }
        if Path(full_path).exists():
            async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
                existing_data = json.loads(await f.read())
                existing_data["message_id"] = message_id
                data = existing_data

        async with aiofiles.open(full_path, "w") as f:
            await f.write(
                json.dumps(data, ensure_ascii=False)
            )

    async def update_chat_history_date(self, chat_id):
        # Updating chat history avaliable for LLM by updating the threshold date
        chat_dir = self.file_path(chat_id)
        full_path = os.path.join(chat_dir, 'chat_data.json')

        async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
        data["chat_history_date"] = time.strftime(
            '%Y-%m-%d %H:%M:%S',
            time.localtime()
        )
        async with aiofiles.open(full_path, "w") as f:
            await f.write(
                json.dumps(data, ensure_ascii=False)
            )
    
    async def insert_message_to_sql(
        self,
        first_name,
        last_name,
        is_bot,
        user_id,
        chat_id,
        message_id,
        send_time,
        message_text,
        username
    ):
        # Saving messages from chat history to SQL DB 
        if self.pool is None:
            self.pool = AsyncConnectionPool(
                f"dbname='customer_bot' user={os.environ.get('DB_USER', '')} password={os.environ.get('DB_PASSWORD', '')} host={os.environ.get('DB_HOST', '')} port={os.environ.get('DB_PORT', '')}",
                min_size=1,
                max_size=10
            )
        async with self.pool.connection() as self.conn:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO chats_history (first_name, last_name, is_bot, user_id, chat_id, message_id, send_time, message_text, username)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (first_name, last_name, is_bot, user_id, chat_id, message_id, send_time, message_text, username))

    async def read_chat_history(
        self,
        chat_id: int,
        message_id: int,
        token: str
    ):
        # Reads the chat history from a telegram server and returns it as a list of messages
        chat_dir = self.file_path(chat_id)
        full_path = os.path.join(chat_dir, 'chat_data.json')
        async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
            chat_history_date = json.loads(await f.read())["chat_history_date"]

        messages = None
        chat_history = []
        service_messages = [
            "Выберете номер вашей заявки ниже 👇",
            "Возвращаюсь в меню...",
            "Секунду..."
        ]
    
        if self.chat_history_client is None:
            self.chat_history_client = Client(
                "memory",
                workdir="./",
                api_id=os.environ.get("TELEGRAM_API_ID", ""),
                api_hash=os.environ.get("TELEGRAM_API_HASH", ""),
                bot_token=token
            )

        self.logger.info(f"Reading chat history for chat id: {chat_id}")
        try:
            await self.chat_history_client.start()
            message_ids = list(range(message_id-199, message_id+1))
            messages = await self.chat_history_client.get_messages(
                chat_id,
                message_ids
            )
        except Exception as e:
            self.logger.error(
                f"Error reading chat history for chat id {chat_id}: {e}"
            )
        finally:
            await self.chat_history_client.stop()

        if messages:
            for message in messages:
                if message.from_user and message.chat.id==chat_id and message.text not in service_messages and message.date > datetime.strptime(chat_history_date, '%Y-%m-%d %H:%M:%S'):
                    if message.from_user.is_bot:
                        chat_history.append(AIMessage(content=message.text))
                    else:
                        chat_history.append(HumanMessage(content=message.text))

            message = messages[-1]
            if message.from_user and message.chat.id==chat_id and message.text not in service_messages and message.date > datetime.strptime(chat_history_date, '%Y-%m-%d %H:%M:%S'):
                first_name = message.from_user.first_name if message.from_user.first_name else None
                last_name = message.from_user.last_name if message.from_user.last_name else None
                username = message.from_user.username if message.from_user.username else None
                is_bot = message.from_user.is_bot
                user_id = message.from_user.id
                msg_id = message.id
                send_time = message.date
                message_text = message.text

                await self.insert_message_to_sql(
                    first_name,
                    last_name,
                    is_bot,
                    user_id,
                    chat_id,
                    msg_id,
                    send_time,
                    message_text,
                    username
                )
        return chat_history[:-1]

    def delete_files(self, chat_id: str):
        # Deletes folder and all its content
        log_path = Path(self.file_path(chat_id))
        if log_path.exists() and log_path.is_dir():
            try:
                shutil.rmtree(log_path)
                self.logger.info(f"Deleted files for chat_id: {chat_id}")
            except Exception as e:
                self.logger.error(
                    f"Error deleting files for chat_id: {chat_id}: {e}"
                )
        else:
            self.logger.info(
                f"No files found for chat_id: {chat_id}, nothing to delete."
            )

    async def save_to_request(
        self,
        chat_id,
        message_text,
        message_type,
        date_override=None,
    ):
        # Saving request item to folder
        self.logger.info(
            f"[{message_type}] Saving request item to request for chat_id: {chat_id}"
        )
        if date_override is None:
            message_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        else:
            message_date = time.strftime(
                "%Y-%m-%d-%H-%M-%S",
                time.localtime(date_override)
            )
        log_file_name = f"{message_type}.json"

        request_dir = self.file_path(chat_id)
        Path(request_dir).mkdir(parents=True, exist_ok=True)
        full_path = os.path.join(request_dir, log_file_name)

        # Adding a comment to the same string
        if Path(full_path).exists() and message_type == "comment":
            async with aiofiles.open(
                full_path,
                "r",
                encoding="utf-8"
            ) as log_file:
                data = await log_file.read()
                existing_data = json.loads(data)
                existing_text = existing_data.get("text", "")
                message_text = existing_text + ". " + message_text

        async with aiofiles.open(full_path, "w") as log_file:
            await log_file.write(
                json.dumps(
                    {
                        "type": message_type,
                        "text": message_text,
                        "date": message_date,
                    },
                    ensure_ascii=False
                )
            )

    async def read_request(self, chat_id: str):
        # Reads request items from a folder and returns it
        request_items = {}
        request_path = self.file_path(chat_id)
        Path(request_path).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Reading request from: {request_path}")

        for item in sorted(os.listdir(request_path)):
            full_path = os.path.join(request_path, item)
            try:
                with open(full_path, "r") as file:
                    message = json.load(file)
                    if message["type"] == "direction":
                        request_items["direction"] = message["text"]
                    elif message["type"] == "phone":
                        request_items["phone"] = message["text"]
                    elif message["type"] == "latitude":
                        request_items["latitude"] = message["text"]
                    elif message["type"] == "longitude":
                        request_items["longitude"] = message["text"]
                    elif message["type"] == "address":
                        request_items["address"] = message["text"]
                    elif message["type"] == "address_line_2":
                        request_items["address_line_2"] = message["text"]
                    elif message["type"] == "date":
                        request_items["date"] = message["text"]
                    elif message["type"] == "comment":
                        request_items["comment"] = message["text"]
                    elif message["type"] == "name":
                        request_items["name"] = message["text"]
            except Exception as e:
                self.logger.error(f"Error reading request file {item}: {e}")
                # Remove problematic file
                os.remove(full_path)
        return request_items
