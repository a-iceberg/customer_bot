import re
from langchain.tools.base import StructuredTool
from file_service import save_to_request


def create_contact_tool(chat_id, message):
    def save_contact():
        save_to_request(
            chat_id,
            "".join(re.findall(r"[+\d]", message["text"])),
            message["message_id"],
            "phone",
        )
        return "Телефон пользователя был получен"

    return StructuredTool.from_function(
        func=save_contact,
        name="Сохранение телефона",
        description="Используйте, когда вам нужно сохранить полученный от пользователя телефон в заявку",
        return_direct=False,
    )
