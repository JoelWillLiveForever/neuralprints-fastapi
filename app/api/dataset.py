from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

import hashlib
import json
import os

from app.models import UploadDatasetRequest

import logging

# Уникальный ключ логгера на этот файл
LOGGER_KEY = "dataset.py"

# Получение глобального логгера
logger = logging.getLogger(LOGGER_KEY)

router = APIRouter()

UPLOAD_DIRECTORY = "./uploaded_datasets"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Эндпоинт для загрузки CSV файла
@router.post(
    "/upload",
    tags=["Dataset"],
    status_code=status.HTTP_201_CREATED, # 201 для успешного создания
    summary="Загрузка набора данных",
    response_description="MD5 хэш успешно загруженного файла",
)
async def upload_dataset(request: UploadDatasetRequest):
    """
    Загружает набор данных в формате JSON с проверкой MD5.

    - **request.filename**: оригинальное имя файла
    - **request.payload**: словарь с полем `csv_data`, содержащим CSV как строку
    - **request.md5**: контрольная сумма содержимого для проверки целостности

    Если файл успешно загружен и целостность подтверждена, возвращается рассчитанный MD5 хэш.
    """
    
    try:
        # Преобразуем payload обратно в строку
        payload_string = json.dumps(request.payload.model_dump(), separators=(",", ":"), ensure_ascii=False)
        
        # Вычисляем MD5 на сервере
        md5_server = hashlib.md5(payload_string.encode('utf-8')).hexdigest()
        
        if md5_server != request.md5_client:
            logger.error(f"Хэши MD5 датасета не совпадают: client = {request.md5_client}, server = {md5_server}")
            raise HTTPException(status_code=400, detail="MD5 mismatch")
        
        csv_data = request.payload.csv_data
        column_types = request.payload.column_types
        dataset_name = request.payload.dataset_name
        
        # Если всё ок — сохраняем CSV данные на диск
        if not csv_data:
            raise HTTPException(400, "CSV data is missing in payload")
        
        # Оставляем только basename (без путей) и cохраняем файл под его оригинальным именем
        safe_filename = os.path.basename(dataset_name)
        file_location = os.path.join(UPLOAD_DIRECTORY, safe_filename)
        
        # Если файл уже существует
        if os.path.exists(file_location):
            # raise HTTPException(409, "Dataset already exists")
            
            # logger.debug(f"Датасет с таким именем уже присутствует: {file_location}")
            return JSONResponse(
                status_code=208,  # 208 Already Reported
                content={"md5_server": md5_server, "message": "Dataset already exists."}
            )
        
        # Сохраняем CSV файл
        with open(file_location, "w", encoding="utf-8") as f:
            f.write(csv_data)
            
        # === Сохраняем мета-файл ===
        meta_filename = f"{os.path.splitext(safe_filename)[0]}.meta.json"  # имя без расширения + .meta.json
        meta_file_location = os.path.join(UPLOAD_DIRECTORY, meta_filename)
            
        meta_data = {
            "column_types": column_types
        }
        
        with open(meta_file_location, "w", encoding="utf-8") as meta_file:
            json.dump(meta_data, meta_file, ensure_ascii=False, indent=2)
            
        logger.info(f"Датасет сохранён по пути: {file_location}")
        return {"md5_server": md5_server}
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке датасета: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Эндпоинт для скачивания CSV файла по MD5 хешу
@router.get(
    "/download/{filename}",
    tags=["Dataset"],
    summary="Скачивание набора данных",
    response_description="CSV файл по оригинальному имени",
)
async def download_dataset(filename: str):
    """
    Скачивает CSV файл по его оригинальному имени.

    - **filename**: название файла (например, `sales_data.csv`)

    Если файл найден, он будет отправлен клиенту для загрузки.
    """
    
    # Защита: basename защищает от попыток выхода за пределы папки
    safe_filename = os.path.basename(filename)
    file_location = os.path.join(UPLOAD_DIRECTORY, safe_filename)

    # Проверяем, существует ли файл
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="Dataset not found")

    return FileResponse(file_location, media_type='application/octet-stream', filename=safe_filename)