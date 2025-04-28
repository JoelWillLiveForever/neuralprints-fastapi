import os
import hashlib

from typing import Optional, Dict

import numpy as np
import pandas as pd

import tensorflow as tf
import keras

from keras import layers
from keras import Sequential

from app.models import ArchitecturePayload

import logging
logger = logging.getLogger(__name__)

MODELS_DIR = "./saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

class ModelTrainer:
    def __init__(self):
        self.current_job: Optional[Dict] = None
        
    def build_model(self, payload: ArchitecturePayload) -> Sequential:
        """Создает модель TensorFlow на основе полученной архитектуры"""
        model = Sequential()
        
        for layer in payload.layers:
            layer_type = layer.type
            data = layer.data.model_dump()
            
            if layer_type == "TF_INPUT_LAYER_NODE":
                # Input layer обрабатывается отдельно
                continue
            
            elif layer_type == "TF_DENSE_LAYER_NODE":
                model.add(keras.layers.Dense(
                    units=data["tf_layer_neurons_count"],
                    activation=["tf_layer_activation_function"],
                    use_bias=data["tf_layer_use_bias"],
                    name=data["tf_layer_name"]
                ))
                
            # Добавьте обработку других типов слоев по аналогии
            
        if not model.layers:
            raise ValueError("Invalid architecture: no layers detected")
        
        return model
    
    def load_dataset(self, md5_hash: str) -> pd.DataFrame:
        """Загружает датасет из файла по MD5 хешу"""
        file_path = os.path.join("./uploaded_datasets", f"{md5_hash}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError("Dataset not found")
        
        return pd.read_csv(file_path)
        
    def preprocess_data(self, data: pd.DataFrame, payload: ArchitecturePayload):
        """Предобработка данных"""
        # Реализуйте вашу логику предобработки
        # Например, нормализацию, разделение на features/target и т.д.
        
        # Пример:
        if payload.enable_dataset_normalization:
            data = (data - data.mean()) / data.std()
            
        return data
    
    def train_model(
        self,
        dataset_md5: str,
        architecture: ArchitecturePayload,
        architecture_md5: str
    ) -> Dict:
        """Основной метод для обучения модели"""
        
        try:
            # Загрузка данных
            raw_data = self.load_dataset(dataset_md5)
            processed_data = self.preprocess_data(raw_data, architecture)
            
            # Разделение данных
            # Реализуйте вашу логику разделения данных на train/test/validation
            # согласно параметрам из architecture.payload
            
            # Построение модели
            model = self.build_model(architecture)
            model.compile(
                optimizer=architecture.optimizer,
                loss=architecture.loss_function,
                metrics=[architecture.quality_metric]
            )
            
            # Обучение модели
            history = model.fit(
                x_train, y_train,
                epochs=architecture.epochs,
                batch_size=architecture.batch_size,
                validation_data=(x_val, y_val)
            )
            
            # Сохранение модели
            model_hash = hashlib.md5(
                f"{architecture_md5}{dataset_md5}".encode()
            ).hexdigest()
            
            model_path = os.path.join(MODELS_DIR, f"{model_hash}.h5")
            model.save(model_path)
            
            return {
                "status": "success",
                "model_hash": model_hash,
                "history": history.history,
                "evaluation": model.evaluate(x_test, y_test)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"status": "error", "message": str(e)}