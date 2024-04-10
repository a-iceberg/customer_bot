import telebot
from langchain.tools.base import StructuredTool


def create_contact_tool(bot, chat_id):
    def send_contact_request():
        keyboard = telebot.types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
        button = telebot.types.KeyboardButton(text="Ваш телефон", request_contact=True)
        keyboard.add(button)
        bot.send_message(
            chat_id, "Укажите ваш телефон через кнопку ниже", reply_markup=keyboard
        )
        return "Контакты пользователя были запрошены"

    return StructuredTool.from_function(
        func=send_contact_request,
        name="Запрос контактов",
        description="Используйте, когда вам нужно запросить контакты пользователя для дальнейшего использования этой информации при создании заявки, чтобы помочь пользователю.",
        return_direct=False,
    )
