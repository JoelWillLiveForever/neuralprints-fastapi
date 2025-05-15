import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# from app.websockets.websocket_manager import websocket_manager
from app.services.training import ModelTrainer

router = APIRouter()

@router.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    trainer = ModelTrainer()

    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "start_training":
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