import telebot
from langchain.tools.base import StructuredTool


def create_location_tool(bot, chat_id):
    def send_location_request():
        keyboard = telebot.types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
        button = telebot.types.KeyboardButton(
            text="Ваше местоположение", request_location=True
        )
        keyboard.add(button)
        bot.send_message(
            chat_id,
            "Укажите ваше местоположение через кнопку ниже",
            reply_markup=keyboard,
        )
        return "Локация пользователя была успешно запрошена"

    return StructuredTool.from_function(
        func=send_location_request,
        name="Request Location",
        description="Используйте, когда вам нужно запросить локацию пользователя для дальнейшего использования этой информации при создании заявки, чтобы помочь пользователю.",
        return_direct=False,
    )
