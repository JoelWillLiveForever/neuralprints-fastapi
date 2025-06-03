import logging

import os
import shutil
import zipfile
import tempfile

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

# Уникальный ключ логгера на этот файл
LOGGER_KEY = "executables.py"

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # api/ → project_root/
# TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
# MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

MODELS_DIR = "./saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

TEMPLATES_DIR = "./app/templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

TMP_ZIP_DIR = "./temp_zips"
os.makedirs(TMP_ZIP_DIR, exist_ok=True)

# Получение глобального логгера
logger = logging.getLogger(LOGGER_KEY)
router = APIRouter()


@router.get(
    "/{model_hash}/download/python",
    tags=['Executables']
)
async def download_inference_script(model_hash: str, n_features: int = Query(..., gt=0)):
    logger.info(f"model_hash = {model_hash}")

    keras_path = os.path.join(MODELS_DIR, f"{model_hash}.keras")
    if not os.path.exists(keras_path):
        raise HTTPException(status_code=404, detail="Keras model file not found")
    
    logger.info(f"keras_path = {keras_path}")

    template_path = os.path.join(TEMPLATES_DIR, "inference_template.py")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=500, detail="Inference template not found")
    
    logger.info(f"template_path = {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        script_template = f.read()
        
    logger.info(f"script_template = {script_template}")

    logger.info(f"==== TEMPLATE BEGIN ====")
    script_content = script_template.format(n_features=n_features)
    logger.info(f"==== TEMPLATE END ====")
    
    logger.info(f"script_content = {script_content}")

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "inference.py")
        keras_copy_path = os.path.join(temp_dir, "model.keras")
        zip_local_path = os.path.join(temp_dir, f"{model_hash}_inference_bundle.zip")

        # Запись файлов
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        with open(keras_path, "rb") as src, open(keras_copy_path, "wb") as dst:
            dst.write(src.read())

        # Архивация
        with zipfile.ZipFile(zip_local_path, "w") as zipf:
            zipf.write(script_path, arcname="inference.py")
            zipf.write(keras_copy_path, arcname="model.keras")

        # Перемещаем zip в постоянную директорию
        final_zip_path = os.path.join(TMP_ZIP_DIR, f"{model_hash}_inference_bundle.zip")
        shutil.copy(zip_local_path, final_zip_path)

    return FileResponse(
        path=final_zip_path,
        filename=f"{model_hash}_inference_bundle.zip",
        media_type="application/zip"
    )
    