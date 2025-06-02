import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import shutil
import tempfile
import logging

# Уникальный ключ логгера на этот файл
LOGGER_KEY = "models.py"
MODELS_DIR = "./saved_models"

# Получение глобального логгера
logger = logging.getLogger(LOGGER_KEY)

router = APIRouter()
    
@router.get(
    "/{model_hash}/download/h5",
    tags=['Models'],
)
async def download_model_h5(model_hash: str):
    logger.info(f"model_hash = {model_hash}")
    
    file_path = os.path.join(MODELS_DIR, f"{model_hash}.h5")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(
        path=file_path,
        filename=f"{model_hash}.h5",
        media_type="application/octet-stream"
    )
    
@router.get(
    "/{model_hash}/download/keras",
    tags=['Models'],
)
async def download_model_keras(model_hash: str):
    logger.info(f"[KERAS] model_hash = {model_hash}")

    file_path = os.path.join(MODELS_DIR, f"{model_hash}.keras")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Keras model not found")

    return FileResponse(
        path=file_path,
        filename=f"{model_hash}.keras",
        media_type="application/octet-stream"
    )
    
@router.get(
    "/{model_hash}/download/savedmodel",
    tags=['Models'],
)
async def download_saved_model(model_hash: str):
    logger.info(f"[SavedModel] model_hash = {model_hash}")

    model_dir = os.path.join(MODELS_DIR, f"{model_hash}_savedmodel")
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail="SavedModel directory not found")

    # Временный файл архива
    temp_dir = tempfile.mkdtemp()
    archive_path = os.path.join(temp_dir, f"{model_hash}_savedmodel.zip")

    # Упаковываем модель в zip
    shutil.make_archive(archive_path.replace('.zip', ''), 'zip', model_dir)

    return FileResponse(
        path=archive_path,
        filename=f"{model_hash}_savedmodel.zip",
        media_type="application/zip"
    )