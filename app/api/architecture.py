import json
import os
import hashlib

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.models import UploadArchitectureRequest, ArchitecturePayload

import logging

# Уникальный ключ логгера на этот файл
LOGGER_KEY = "architecture.py"

# Получение глобального логгера
logger = logging.getLogger(LOGGER_KEY)

router = APIRouter()

# ПАМЯТЬ СЕРВЕРА
# architecture = None

# Путь для хранения файлов архитектур
ARCHITECTURE_STORAGE_PATH = "./uploaded_architectures"

# Убедимся, что папка для хранения архитектур существует
os.makedirs(ARCHITECTURE_STORAGE_PATH, exist_ok=True)

@router.post(
    "/upload", 
    tags=['Architecture'],
    status_code=status.HTTP_201_CREATED, # 201 для успешного создания
    summary="Получение архитектуры", 
    description="Принимает архитектуру модели и сохраняет её на сервере."
)
async def receive_architecture(request: UploadArchitectureRequest):
    """
    Принимает архитектуру модели в формате JSON, сохраняет её в файл с хэш-именем и генерирует MD5 хеш.
    """
    
    try:        
        # Получаем словарь из Pydantic модели
        payload_dict = request.payload.model_dump()
        print(f'\n\nТекущая архитектура: ${payload_dict}\n\n')
        
        # Сериализуем с сортировкой ключей и без пробелов
        payload_string = json.dumps(
            payload_dict,
            separators=(",", ":"),
            ensure_ascii=False,
            sort_keys=True  # Добавляем сортировку ключей
        )
        
        # Создаем MD5 хеш для архитектуры
        md5_server = hashlib.md5(payload_string.encode('utf-8')).hexdigest()
        
        if md5_server != request.md5_client:
            logger.error(f"Хэши MD5 архитектуры ИИ не совпадают: client = {request.md5_client}, server = {md5_server}, content = {payload_string}")
            raise HTTPException(status_code=400, detail="MD5 mismatch")
    
        # Генерируем путь для файла, основываясь на MD5 хеше
        architecture_file_path = os.path.join(ARCHITECTURE_STORAGE_PATH, f"{md5_server}.meta.json")
        
        # Если файл уже существует
        if os.path.exists(architecture_file_path):
            # raise HTTPException(409, "Dataset already exists")
            
            # logger.info(f"This architecture already exists: {architecture_file_path}")
            return JSONResponse(
                status_code=208,  # 208 Already Reported
                content={"md5_server": md5_server, "message": "Architecture already exists."}
            )
        
        # Сохраняем архитектуру в файл
        with open(architecture_file_path, "w") as f:
            # Используем ту же сериализацию, что и для хэша
            f.write(payload_string)
            
        return {
            "md5_server": md5_server
        }
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке архитектуры ИИ: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get(
    "/download", 
    tags=['Architecture'],
    response_model=ArchitecturePayload, 
    summary="Отправка архитектуры", 
    description="Возвращает текущую архитектуру модели."
)
async def send_architecture(md5_hash: str):
    """
    Возвращает сохраненную архитектуру модели по хэшу.

    Если архитектура не была получена, возвращает ошибку.
    """
    
    # global architecture
    # if architecture is None:
    #     raise HTTPException(status_code=404, detail='Error: architecture is "None"')
    
    architecture_file_path = os.path.join(ARCHITECTURE_STORAGE_PATH, f"{md5_hash}.meta.json")
    
    if not os.path.exists(architecture_file_path):
        raise HTTPException(status_code=404, detail="Error: architecture not found")
    
    with open(architecture_file_path, "r") as f:
        architecture_data = json.load(f)
    
    return architecture_data