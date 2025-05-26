import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# from app.websockets.websocket_manager import websocket_manager
from app.services.training import ModelTrainer

import logging

# Уникальный ключ логгера на этот файл
LOGGER_KEY = "sockets.py"

# Получение глобального логгера
logger = logging.getLogger(LOGGER_KEY)

router = APIRouter()

@router.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    trainer = ModelTrainer()

    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "start_training":
                logger.info(f'dataset_name = {data["dataset_name"]}')
                logger.info(f'architecture_hash = {data["architecture_hash"]}')
                
                await trainer.train_model_ws(
                    dataset_name=data["dataset_name"],
                    architecture_name=data["architecture_hash"],
                    websocket=websocket
                )
                
    except WebSocketDisconnect:
        print("Клиент отключился")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })