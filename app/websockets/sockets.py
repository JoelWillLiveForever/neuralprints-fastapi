import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# from app.websockets.websocket_manager import websocket_manager
from app.services.training import ModelTrainer

router = APIRouter()

@router.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "start_training":
                dataset_name = data["dataset_name"]
                architecture_hash = data["architecture_hash"]

                trainer = ModelTrainer()

                await trainer.train_model_ws(
                    dataset_name=dataset_name,
                    architecture_name=architecture_hash,
                    websocket=websocket # Прямо передаём WebSocket этого клиента
                )

    except WebSocketDisconnect:
        print("Клиент отключился")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })