from pydantic import BaseModel
from typing import List, Union, Literal

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
    
    
# === МОДЕЛЬ ЗАПРОСА НА ПОЛУЧЕНИЕ CSV ДАТАСЕТА ===
class UploadDatasetPayload(BaseModel):
    dataset_name: str
    column_types: List[str]
    csv_data: str # строка с данными CSV

class UploadDatasetRequest(BaseModel):
    md5: str
    payload: UploadDatasetPayload