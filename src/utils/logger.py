"""
日志配置模块

该模块负责配置应用程序的日志系统，包括文件日志和控制台日志。
"""

import logging
from logging.handlers import TimedRotatingFileHandler

# 设置日志格式
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')

# 创建TimedRotatingFileHandler，每天滚动一次，保留最近7天的日志
log_file = "server.log"
handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7, encoding='utf-8')
handler.setFormatter(formatter)

# 设置日志级别
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# 可选：添加控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)