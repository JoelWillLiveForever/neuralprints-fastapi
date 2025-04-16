import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List, Union, Literal

import hashlib
import json

app = FastAPI()

origins = [
    "http://localhost:5173"
    # ""
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === МОДЕЛИ ДЛЯ СЛОЁВ ===
class InputLayerData(BaseModel):
    tf_layer_name: str
    tf_layer_neurons_count: int

class DenseLayerData(BaseModel):
    tf_layer_name: str
    tf_layer_neurons_count: int
    tf_layer_activation_function: str
    tf_layer_use_bias: bool

class DropoutLayerData(BaseModel):
    tf_layer_name: str
    tf_layer_strength: float

class GaussianDropoutLayerData(BaseModel):
    tf_layer_name: str
    tf_layer_strength: float

class GaussianNoiseLayerData(BaseModel):
    tf_layer_name: str
    tf_layer_stddev: float

class FlattenLayerData(BaseModel):
    pass

class Conv2DLayerData(BaseModel):
    tf_layer_name: str
    lt_layer_filters_count: int
    tf_layer_kernel_size: tuple[int, int]
    tf_layer_activation_function: str
    tf_layer_use_bias: bool


# === ОБЩАЯ СХЕМА СЛОЯ С УЧЁТОМ type ===
class InputLayer(BaseModel):
    type: Literal["TF_INPUT_LAYER_NODE"]
    data: InputLayerData

class DenseLayer(BaseModel):
    type: Literal["TF_DENSE_LAYER_NODE"]
    data: DenseLayerData

class DropoutLayer(BaseModel):
    type: Literal["TF_DROPOUT_LAYER_NODE"]
    data: DropoutLayerData

class GaussianDropoutLayer(BaseModel):
    type: Literal["TF_GAUSSIAN_DROPOUT_LAYER_NODE"]
    data: GaussianDropoutLayerData

class GaussianNoiseLayer(BaseModel):
    type: Literal["TF_GAUSSIAN_NOISE_LAYER_NODE"]
    data: GaussianNoiseLayerData

class FlattenLayer(BaseModel):
    type: Literal["TF_FLATTEN_LAYER_NODE"]
    data: FlattenLayerData

class Conv2DLayer(BaseModel):
    type: Literal["TF_CONV_2D_LAYER_NODE"]
    data: Conv2DLayerData


# === UNION ВСЕХ СЛОЁВ ===
Layer = Union[InputLayer, DenseLayer, DropoutLayer, GaussianDropoutLayer, GaussianNoiseLayer, FlattenLayer, Conv2DLayer]


# === МОДЕЛЬ ВСЕЙ ОТПРАВКИ ===
class ArchitecturePayload(BaseModel):
    layers: List[Layer]
    train_split: float
    test_split: float
    validation_split: float
    loss_function: str
    optimizer: str
    quality_metric: str
    epochs: int
    batch_size: int
    enable_dataset_normalization: bool


# === ПАМЯТЬ СЕРВЕРА ===
architecture: ArchitecturePayload | None = None


# === POST методы (отправка данных на сервер) ===
@app.post("/architecture")
async def receive_architecture(payload: ArchitecturePayload):
    if payload is None:
        raise HTTPException(status_code=404, detail='Received architecture is \"None\"')
    
    global architecture
    architecture = payload
    
    for layer in architecture.layers:
        print(f"Тип слоя: {layer.type}")
        print(f"Данные слоя: {layer.data} \n\n")

    # Преобразуем payload в JSON-строку (для хэширования)
    payload_json = json.dumps(payload.model_dump(), sort_keys=True)  # Сортируем ключи для стабильности хэша
    md5_hash = hashlib.md5(payload_json.encode('utf-8')).hexdigest()

    return {
        "status": "ok",
        "message": "Architecture received successfully",
        "md5": md5_hash
    }


# === GET методы (получение данных с сервера) ===
@app.get("/architecture", response_model=ArchitecturePayload)
async def send_architecture():
    global architecture

    if architecture is None:
        raise HTTPException(status_code=404, detail='Architecture is \"None\"')

    return architecture


# === MAIN метод ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# class Fruit(BaseModel):
#     name: str

# class Fruits(BaseModel):
#     fruits: List[Fruit]



# memory_db = {"fruits": []}

# @app.get("/fruits", response_model=Fruits)
# def get_fruits():
#     return Fruits(fruits=memory_db["fruits"])

# @app.post("/fruits")
# def add_fruit(fruit: Fruit):
#     memory_db["fruits"].append(fruit)
#     return fruit