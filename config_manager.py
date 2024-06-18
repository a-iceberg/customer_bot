import json


class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as config_file:
            return json.load(config_file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def save_config(self):
        with open(self.config_path, 'w', encoding='utf-8') as config_file:
            json.dump(self.config, config_file, ensure_ascii=False, indent=4)