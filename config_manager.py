import json


class ConfigManager:
    def __init__(self, config_path, logger):
        self.logger = logger
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
        self.config = self.load_config()
    
    def delete(self, key):
        self.config = self.load_config()
        if key in self.config:
            del self.config[key]
            self.save_config()
        else:
            self.logger.warnig("Nothing to delete")
        self.config = self.load_config()

    def save_config(self):
        with open(self.config_path, 'w', encoding='utf-8') as config_file:
            json.dump(self.config, config_file, ensure_ascii=False, indent=4)