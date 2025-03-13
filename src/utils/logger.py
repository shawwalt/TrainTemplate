import logging
import os

class Logger:
    def __init__(self, log_name="app_log", log_dir="./meta/logs", log_level=logging.DEBUG):
        """
        初始化日志记录器

        :param log_name: 日志文件名（不带扩展名）
        :param log_dir: 日志文件存放目录
        :param log_level: 日志级别
        """
        self.log_name = log_name
        self.log_dir = log_dir
        self.log_level = log_level

        # 创建日志存储目录，如果目录不存在则创建
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 创建日志对象
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(self.log_level)

        # 创建日志文件的 handler
        log_file = os.path.join(self.log_dir, f"{self.log_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)

        # 创建日志格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 创建控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 控制台只显示 INFO 级别以上的日志
        console_handler.setFormatter(formatter)

        # 将 handlers 添加到 logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        """
        记录 DEBUG 级别日志
        """
        self.logger.debug(message)

    def info(self, message):
        """
        记录 INFO 级别日志
        """
        self.logger.info(message)

    def warning(self, message):
        """
        记录 WARNING 级别日志
        """
        self.logger.warning(message)

    def error(self, message):
        """
        记录 ERROR 级别日志
        """
        self.logger.error(message)

    def critical(self, message):
        """
        记录 CRITICAL 级别日志
        """
        self.logger.critical(message)

    def set_level(self, log_level):
        """
        设置日志级别
        """
        self.logger.setLevel(log_level)

    def remove_handlers(self):
        """
        移除所有 handlers
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
