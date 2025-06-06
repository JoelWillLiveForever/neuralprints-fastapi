from contextlib import asynccontextmanager
import os

# Environment variables
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import nest_asyncio  # Импорт библиотеки для работы с вложенными циклами событий

import uvicorn
from fastapi import FastAPI

from app.config import configure_cors  # Импортируем конфигурацию CORS
from app.api import dataset, architecture, executables, tensorflow, models  # Импортируем обработку эндпоинтов для архитектуры и датасета

# from app.websockets.websocket_manager import websocket_manager
from app.websockets import sockets

import logging

# Применяем патч для вложенных циклов событий
nest_asyncio.apply()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Добавлен вывод имени логгера
    level=logging.DEBUG,  # Уровень логирования
    handlers=[
        # logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler('app.log')  # Запись в файл
    ]
)

# Уникальный ключ логгера на этот файл
LOGGER_KEY = "main.py"

# Получение глобального логгера
logger = logging.getLogger(LOGGER_KEY)

# Логгер подключен в Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код при старте приложения
    logger.info("Server starting up")
    yield
    # Код при остановке приложения
    logger.info("Server shutting down")

# Инициализируем приложение
app = FastAPI(
    title="NeuralPrints API",
    description="API для обучения нейросетевых моделей",
    version="1.0.0",
    contact={
        "name": "Vladimir (aBLeeWeeAN)",
        "email": "mail.jorey@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
    # Группировка эндпоинтов
    openapi_tags=[
        {
            "name": "Dataset",
            "description": "Работа с датасетом",
        },
        {
            "name": "Architecture",
            "description": "Управление архитектурой моделей"
        },
        # {
        #     "name": "TensorFlow",
        #     "description": "Управление процессом обучения моделей"
        # },
        {
            "name": "Models",
            "description": "Управление процессом экспорта моделей"
        },
        {
            "name": "Executables",
            "description": "Управление процессом экспорта исполняемых файлов"
        },
    ],
    lifespan=lifespan,
)

# Настроить CORS (если необходимо)
configure_cors(app)

# Сокеты
app.include_router(sockets.router)

# Регистрируем эндпоинты
app.include_router(dataset.router, prefix="/api/dataset")
app.include_router(architecture.router, prefix="/api/architecture")
# app.include_router(tensorflow.router, prefix="/api/tensorflow")
app.include_router(models.router, prefix="/api/models")
app.include_router(executables.router, prefix="/api/executables")

# === MAIN метод ===
def __main__():
    logger.debug('Вызван метод __main__()')
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, workers=1, loop="asyncio", timeout_keep_alive=300)

if __name__ == "__main__":
    __main__()