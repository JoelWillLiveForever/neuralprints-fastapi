from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import hashlib

import asyncio

import time
from typing import Optional, Dict

import numpy as np
import pandas as pd

# import tensorflow as tf
import keras

# from keras import layers
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

from app.models import ArchitecturePayload
# from app.websockets.websocket_manager import WebSocketManager

import logging

from fastapi import WebSocket

# Уникальный ключ логгера на этот файл
LOGGER_KEY = "training.py"

# Получение глобального логгера
logger = logging.getLogger(LOGGER_KEY)

MODELS_DIR = "./saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

class TrainingProgressCallback(keras.callbacks.Callback):
    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop, all_epochs: int, metric_name: str):
        super().__init__()
        self.websocket = websocket
        self.loop = loop
        self.all_epochs = all_epochs
        self.metric_name = metric_name
        
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        # Запоминаем момент начала эпохи
        self.epoch_start_time = time.monotonic()
        
    def on_epoch_end(self, epoch, logs=None):
        # 1) Получаем текущий момент в ISO‑формате UTC
        now = datetime.now(timezone.utc).astimezone()  # локальный часовой пояс
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # 2) Вычисляем timing_str
        # Сколько длилась эпоха
        elapsed = time.monotonic() - self.epoch_start_time  # seconds as float
        
        # Число шагов за эпоху можно взять из params
        num_steps = self.params.get('steps', None)
        if num_steps is None:
            # fallback: 
            num_steps = logs.get('size', 0) # self.params.get('batch_size', 1)
            
        # Время на один шаг (batch)
        per_step = elapsed / num_steps if num_steps else 0  # seconds
        
        # Форматируем в строку вида "0s 2ms/step"
        sec = int(elapsed)
        ms_per_step = int(per_step * 1000)
        timing_str = f"{sec}s {ms_per_step}ms/step"
        
        # 3) Формируем сообщение лога с меткой времени
        asyncio.run_coroutine_threadsafe(
            self.websocket.send_json({
                "type": "log", 
                "log_message": (
                    f"[{timestamp}] Epoch {epoch+1}/{self.all_epochs}\n{timing_str} ——— "
                    f"{self.metric_name}: {logs.get(self.metric_name):.4f} — "
                    f"loss: {logs.get('loss'):.4f} — "
                    f"val_{self.metric_name}: {logs.get(f'val_{self.metric_name}'):.4f} — "
                    f"val_loss: {logs.get('val_loss'):.4f}"
                )
            }),
            self.loop
        )
        
        # основная инфа
        message = {
            "type": "training_update",
            "current_epoch": epoch + 1,
            "training_loss": logs.get("loss"),
            "validation_loss": logs.get("val_loss"),
            "training_user_metric": logs.get(self.metric_name),
            "validation_user_metric": logs.get(f"val_{self.metric_name}"),
        }
        
        # Запускаем корутину отправки в основном event-loop
        fit = asyncio.run_coroutine_threadsafe(
            self.websocket.send_json(message),
            self.loop
        )
        
        try:
            fit.result(timeout=1)
        except Exception as e:
            logger.error(f"WS send failed: {e}")

class ModelTrainer:
    def __init__(self):
        self.current_job: Optional[Dict] = None
        
    def load_model(self, model_name: str) -> keras.Model:
        """Загружает ранее сохранённую модель по имени"""
        model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found")
        
        model = keras.models.load_model(model_path)
        return model
        
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
                    name=data['tf_layer_name'],
                    units=data['tf_layer_neurons_count'],
                    activation=data['tf_layer_activation_function'],
                    use_bias=data['tf_layer_use_bias']
                ))
                
            elif layer_type == "TF_DROPOUT_LAYER_NODE":
                model.add(keras.layers.Dropout(
                    name=data["tf_layer_name"],
                    rate=data['tf_layer_strength']
                ))
                
            elif layer_type == "TF_GAUSSIAN_DROPOUT_LAYER_NODE":
                model.add(keras.layers.GaussianDropout(
                    name=data['tf_layer_name'],
                    rate=data['tf_layer_strength']
                ))
                
            elif layer_type == "TF_GAUSSIAN_NOISE_LAYER_NODE":
                model.add(keras.layers.GaussianNoise(
                    name=data['tf_layer_name'],
                    stddev=data['tf_layer_stddev']
                ))
                
            elif layer_type == "TF_CONV_2D_LAYER_NODE":
                model.add(keras.layers.Conv2D(
                    name=data['tf_layer_name'],
                    filters=data['tf_layer_filters_count'],
                    kernel_size=data['tf_layer_kernel_size'],
                    activation=data['tf_layer_activation_function'],
                    use_bias=data['tf_layer_use_bias']
                ))
                
            elif layer_type == "TF_FLATTEN_LAYER_NODE":
                model.add(keras.layers.Flatten())
                
            # Добавьте обработку других типов слоев по аналогии
            
        if not model.layers:
            raise ValueError("Invalid architecture: no layers detected")
        
        return model
    
    def load_dataset_and_meta(self, dataset_name: str) -> tuple[pd.DataFrame, list[str]]:
        """Загружает датасет и информацию о типах колонок"""
        
        dataset_path = os.path.join("./uploaded_datasets", f"{dataset_name}.csv")
        meta_path = os.path.join("./uploaded_datasets", f"{dataset_name}.meta.json")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("Dataset not found")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Meta information (column types) not found")
        
        # Загружаем сам датасет
        dataframe = pd.read_csv(dataset_path)
         
        # Загружаем типы колонок
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
         
        column_types = meta_info["column_types"]
        
        return dataframe, column_types
    
    def load_architecture(self, architecture_name: str) -> ArchitecturePayload:
        """Загружает архитектуру модели из мета-файла."""
        
        # Путь к мета-файлу архитектуры
        architecture_path = os.path.join("./uploaded_architectures", f"{architecture_name}.meta.json")
        
        # Проверка наличия мета-файла
        if not os.path.exists(architecture_path):
            raise FileNotFoundError(f"Architecture {architecture_name} not found")
        
        # Загружаем архитектуру из файла
        with open(architecture_path, "r", encoding="utf-8") as f:
            architecture_data = json.load(f)
        
        # Преобразуем данные из мета-файла в объект ArchitecturePayload
        # Для этого предполагается, что в мета-файле есть необходимые параметры для создания объекта
        architecture_payload = ArchitecturePayload(**architecture_data)
        
        return architecture_payload
    
    def prepare_features_and_target(self, df: pd.DataFrame, column_types: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        """Разделяет датафрейм на признаки (X) и целевую переменную (y) на основе типов колонок"""
        
        if len(df.columns) != len(column_types):
            raise ValueError("Number of column types does not match number of dataset columns")
        
        feature_columns = []
        target_column = None
        
        for col_name, col_type in zip(df.columns, column_types):
            if col_type == "feature":
                feature_columns.append(col_name)
            elif col_type == "target":
                if target_column is not None:
                    raise ValueError("Multiple target columns detected — only one target column is supported")
                target_column = col_name
            # if ignored — ничего не добавляем
            
        if not feature_columns:
            raise ValueError("No feature columns found")
        if target_column is None:
            raise ValueError("No target column found")
        
        X = df[feature_columns]
        y = df[target_column]
        
        return X, y
      
    def split_dataset(self, X: pd.DataFrame, y: pd.Series, train_size: float = 0.7, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Делит данные на train/validation/test"""
        
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("train_size + val_size + test_size must be 1.0")
        
        # Сначала делим на train и temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )
        
        # Теперь делим temp на val и test
        val_ratio = val_size / (val_size + test_size)  # внутри "temp" какая доля идет в валидацию
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=random_state
        )
        
        return (
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            X_test.values, y_test.values
        )
       
    def preprocess_data(self, data: pd.DataFrame, payload: ArchitecturePayload, column_types: list[str]) -> pd.DataFrame:
        """Предобработка данных"""
        
        # # Нормализация данных (если включено в настройках)
        # if payload.enable_dataset_normalization:
        #     data = (data - data.mean()) / data.std()
        #     logger.info("Dataset normalized.")
            
        # # Обработка пропусков, например, заменой на среднее значение
        # if payload.enable_fill_missing_values:
        #     data = data.fillna(data.mean())
        #     logger.info("Missing values filled.")
            
        # # Преобразование категориальных признаков в числовые
        # if payload.enable_category_encoding:
        #     categorical_cols = data.select_dtypes(include=["object"]).columns
        #     data = pd.get_dummies(data, columns=categorical_cols)
        #     logger.info("Categorical variables encoded.")
            
        # # Другие возможные преобразования
        # # Например, если нужно выбрать только определённые признаки
        # if payload.selected_features:
        #     data = data[payload.selected_features]
        #     logger.info(f"Selected features: {payload.selected_features}")
            
        # # Если нужны дополнительные преобразования (например, логарифмирование)
        # if payload.enable_log_transform:
        #     data = data.apply(np.log1p)  # Применение логарифма с добавлением 1
        #     logger.info("Log transformation applied.")
            
        # # Создаем словарь для замены текстовых значений на числовые автоматически
        # target_values = data['target'].unique()  # Получаем уникальные значения
        # target_mapping = {value: idx for idx, value in enumerate(target_values)}  # Сопоставляем значения с индексами
        
        # # Заменяем значения в столбце 'target' на соответствующие числовые ключи
        # data['target'] = data['target'].map(target_mapping)
            
        # return data
    
        # Пройдем по всем столбцам и найдем столбец с типом 'target'
        for idx, column_type in enumerate(column_types):
            if column_type == 'target':
                target_column = data.columns[idx]  # Получаем имя столбца по индексу
                
                # Создаем словарь для замены текстовых значений на числовые
                target_values = data[target_column].unique()  # Получаем уникальные значения в столбце target
                target_mapping = {value: idx for idx, value in enumerate(target_values)}  # Создаем mapping
                
                # Заменяем значения в столбце 'target' на соответствующие числовые ключи
                data[target_column] = data[target_column].map(target_mapping)
                break  # Так как у нас только один столбец с 'target', можно выйти из цикла
        
        # Преобразуем все столбцы в числовые типы (например, float64), чтобы избежать ошибок с типами
        data = data.apply(pd.to_numeric, errors='coerce')  # Преобразуем все столбцы в числовые, нечисловые заменяются на NaN
        data = data.dropna()  # Удаляем строки с NaN значениями (если они есть)
        
        return data
    
    def train_model(
        self,
        dataset_name: str,
        architecture_name: str
    ) -> Dict:
        logger.debug(f"Вызван метод train_model(): dataset_name = {dataset_name}, architecture_name = {architecture_name}")
        
        try:
            # === 1. Загрузка датасета и метаинформации ===
            data, column_types = self.load_dataset_and_meta(dataset_name)
            logger.info(f"Датасет и типы его столбцов загружены: {dataset_name}")
            
            architecture = self.load_architecture(architecture_name)
            logger.info(f"Архитектура модели загружена: {architecture_name}")
            
            # === 2. Предобработка данных ===
            processed_data = self.preprocess_data(data, architecture, column_types)
            
            # Сохраняем DataFrame в CSV
            # processed_data.to_csv('debug.csv', index=False)  # index=False не записывает индекс в файл
            
            # === 3. Разделение на X (признаки) и y (цель) ===
            X, y = self.prepare_features_and_target(processed_data, column_types)
            
            # X.to_csv('X.csv', index=False)
            # y.to_csv('y.csv', index=False)
            
            # === 4. Разбивка на Train/Validation/Test ===
            (
                x_train, y_train,
                x_val, y_val,
                x_test, y_test
            ) = self.split_dataset(
                X, y,
                train_size=architecture.train_split,
                val_size=architecture.validation_split,
                test_size=architecture.test_split
            )
            logger.info(f"Датасет разбит на train/val/test выборки")
            
            # === 5. Построение модели ===
            model = self.build_model(architecture)
            logger.info(f"Завершена сборка модели ИИ")
            
            # === 6. Компиляция модели ===
            model.compile(
                optimizer=architecture.optimizer,
                loss=architecture.loss_function,
                metrics=[architecture.quality_metric]
            )
            logger.info(f"Завершена компиляция модели ИИ")
            
            # === 7. Кастомные коллбэки ===
            # my_callbacks = []
            # if websocket_manager.active_connections:
            #     my_callbacks.append(TrainingProgressCallback())
            # logger.info(f"Завершена настройка кастомного коллбэка: my_callbacks = {my_callbacks}")          
            
            # === 8. Обучение модели ===
            logger.info(f"Начато обучение модели ИИ")
            
            history = model.fit(
                x_train, y_train,
                epochs=architecture.epochs,
                batch_size=architecture.batch_size,
                validation_data=(x_val, y_val)
                # callbacks=[TrainingProgressCallback()]
            )
            
            logger.info(f"Зарешено обучение модели ИИ")
            
            # === 9. Оценка модели на тестовых данных ===
            evaluation = model.evaluate(x_test, y_test)
            logger.info(f"Завершен анализ качества предсказаний модели ИИ")
            
            # === 10. Сохранение модели ===
            model_hash = hashlib.md5(f"{architecture_name}{dataset_name}".encode()).hexdigest()
            model_path = os.path.join(MODELS_DIR, f"{model_hash}.h5")
            model.save(model_path)
            logger.info(f"Модель ИИ сохранена по пути: {model_path}")
            
            # === 11. Возврат результата ===
            return {
                "status": "success",
                "model_hash": model_hash,
                "history": history.history,
                "evaluation": evaluation
            }
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели в методе train_model(): {str(e)}")
            
            # websocket_manager.broadcast(json.dumps({
            #     "type": "error",
            #     "message": str(e)
            # }))
            return {"status": "error", "message": str(e)}
    
    async def train_model_ws(
        self,
        dataset_name: str,
        architecture_name: str,
        websocket: WebSocket
    ) -> Dict:
        logger.debug(f"Вызван метод train_model_ws(): dataset_name = {dataset_name}, architecture_name = {architecture_name}, websocket = {websocket}")
        await websocket.send_json({"type": "training_started"})
        
        try:
            # === 1. Загрузка датасета и метаинформации ===
            data, column_types = self.load_dataset_and_meta(dataset_name)
            logger.info(f"Датасет и типы его столбцов загружены: {dataset_name}")
            
            architecture = self.load_architecture(architecture_name)
            logger.info(f"Архитектура модели загружена: {architecture_name}")
            
            # === 2. Предобработка данных ===
            processed_data = self.preprocess_data(data, architecture, column_types)
            
            # Сохраняем DataFrame в CSV
            # processed_data.to_csv('debug.csv', index=False)  # index=False не записывает индекс в файл
            
            # === 3. Разделение на X (признаки) и y (цель) ===
            X, y = self.prepare_features_and_target(processed_data, column_types)
            
            # X.to_csv('X.csv', index=False)
            # y.to_csv('y.csv', index=False)
            
            # === 4. Разбивка на Train/Validation/Test ===
            (
                x_train, y_train,
                x_val, y_val,
                x_test, y_test
            ) = self.split_dataset(
                X, y,
                train_size=architecture.train_split,
                val_size=architecture.validation_split,
                test_size=architecture.test_split
            )
            logger.info(f"Датасет разбит на train/val/test выборки")
            
            # === 5. Построение модели ===
            model = self.build_model(architecture)
            logger.info(f"Завершена сборка модели ИИ")
            
            # === 6. Компиляция модели ===
            model.compile(
                optimizer=architecture.optimizer,
                loss=architecture.loss_function,
                metrics=[architecture.quality_metric]
            )
            logger.info(f"Завершена компиляция модели ИИ")
            
            # === 7. Обучение модели ===
            logger.info(f"Начато обучение модели ИИ")
            
            loop = asyncio.get_running_loop()
            callback = TrainingProgressCallback(
                websocket=websocket,
                loop=loop,
                all_epochs = architecture.epochs,
                metric_name=architecture.quality_metric
            )
            
            def _fit():
                model.fit(
                    x_train, y_train,
                    epochs=architecture.epochs,
                    batch_size=architecture.batch_size,
                    validation_data=(x_val, y_val),
                    callbacks=[callback]
                )
                
            # Запускаем учёбу в другом потоке
            with ThreadPoolExecutor() as pool:
                await loop.run_in_executor(pool, _fit)

            # Отправка финального сообщения
            # await websocket.send_json({"type": "training_complete"})
            
            logger.info(f"Завершено обучение модели ИИ")
            
            # === 8. Оценка модели ===
            logger.info("Начата оценка модели на тестовых данных")
            
            # --- 8.1. Предсказания на train, test, val ---
            train_pred_probs = model.predict(x_train)
            test_pred_probs = model.predict(x_test)
            val_pred_probs = model.predict(x_val)
            
            # --- 8.2. Преобразование вероятностей в метки ---
            train_preds = (train_pred_probs > 0.5).astype("int32").flatten()
            test_preds = (test_pred_probs > 0.5).astype("int32").flatten()
            val_preds = (val_pred_probs > 0.5).astype("int32").flatten()
            
            # --- 8.3. Истинные метки (если one-hot, возьми [:, 0]) ---
            y_train_true = y_train.flatten()
            y_test_true = y_test.flatten()
            y_val_true = y_val.flatten()
            
            # --- 8.4. Loss и Accuracy (Keras .evaluate) ---
            train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
            
            # --- 8.5. Метрики через sklearn ---
            precision = precision_score(y_test_true, test_preds)
            recall = recall_score(y_test_true, test_preds)
            f1 = f1_score(y_test_true, test_preds)
            auc = roc_auc_score(y_test_true, test_pred_probs)  # Используем именно вероятности
            
            logger.info(f"Завершена оценка модели на тестовых данных")
            
            # Отправка результатов оценки клиенту
            await websocket.send_json({
                "type": "training_complete",
                
                "final_train_accuracy": train_acc,
                "final_test_accuracy": test_acc,
                "final_validation_accuracy": val_acc,
                
                "final_train_loss": train_loss,
                "final_test_loss": test_loss,
                "final_validation_loss": val_loss,
                
                "final_precision": precision,
                "final_recall": recall,
                "final_f1_score": f1,
                "final_auc_roc": auc
            })
            
            # === 9. Сохранение модели ===
            model_hash = hashlib.md5(f"{architecture_name}{dataset_name}".encode()).hexdigest()
            model_path = os.path.join(MODELS_DIR, f"{model_hash}.h5")
            model.save(model_path)
            
            logger.info(f"Модель ИИ сохранена по пути: {model_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели в методе train_model(): {str(e)}")
            
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        
        
        
        
        