from fastapi import APIRouter, HTTPException
from app.models import ArchitecturePayload
from app.utils import calculate_md5_from_bytes

router = APIRouter()

# ПАМЯТЬ СЕРВЕРА
architecture = None

@router.post("/architecture")
async def receive_architecture(payload: ArchitecturePayload):
    global architecture
    if payload is None:
        raise HTTPException(status_code=404, detail='Error: received architecture is "None"')
    
    architecture = payload

    # Преобразуем payload в JSON-строку с использованием model_dump_json
    payload_json = payload.model_dump_json(sort_keys=True)  # Используем новый метод model_dump_json

    # Используем функцию для вычисления MD5
    md5_hash = calculate_md5_from_bytes(payload_json.encode('utf-8'))

    return {
        "status": "ok",
        "message": "Architecture received successfully",
        "md5": md5_hash
    }

@router.get("/architecture", response_model=ArchitecturePayload)
async def send_architecture():
    global architecture
    if architecture is None:
        raise HTTPException(status_code=404, detail='Error: architecture is "None"')

    return architecture