import uvicorn
from fastapi import FastAPI
from app.config import configure_cors  # Импортируем конфигурацию CORS
from app.api import dataset, architecture, tensorflow  # Импортируем обработку эндпоинтов для архитектуры и датасета

# Инициализируем приложение
app = FastAPI(
    title="NeuralPrints API",
    description="API для обучения нейросетевых моделей",
    version="0.0.1",
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
        {
            "name": "TensorFlow",
            "description": "Управление процессом обучения моделей"
        },
    ]
)

# Настроить CORS (если необходимо)
configure_cors(app)

# Регистрируем эндпоинты
app.include_router(dataset.router, prefix="/api/dataset")
app.include_router(architecture.router, prefix="/api/architecture")
app.include_router(tensorflow.router, prefix="/api/tensorflow")

# === MAIN метод ===
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)