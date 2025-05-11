# from fastapi import WebSocket, WebSocketDisconnect
# from typing import List

# class WebSocketManager:
#     def __init__(self):
#         self.active_connections: List[WebSocket] = []

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)

#     async def disconnect(self, websocket: WebSocket):
#         self.active_connections.remove(websocket)
#         await websocket.close()

#     async def send_personal_message(self, message: str, websocket: WebSocket):
#         await websocket.send_text(message)

#     async def broadcast(self, message: str):
#         for connection in self.active_connections:
#             await connection.send_text(message)

# websocket_manager = WebSocketManager()

# from fastapi import WebSocket
# from typing import List
# import asyncio

# class WebSocketManager:
#     def __init__(self):
#         self.active_connections: List[WebSocket] = []

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)

#     def disconnect(self, websocket: WebSocket):
#         if websocket in self.active_connections:
#             self.active_connections.remove(websocket)

#     async def send_personal_message(self, message: str, websocket: WebSocket):
#         await websocket.send_text(message)
        
#     async def broadcast_async(self, message: str):
#         to_remove = []
#         for connection in self.active_connections:
#             try:
#                 await connection.send_text(message)
#             except Exception:
#                 to_remove.append(connection)
#         for conn in to_remove:
#             self.disconnect(conn)

#     def broadcast(self, message: str):
#         try:
#             loop = asyncio.get_running_loop()
#         except RuntimeError:
#             loop = asyncio.get_event_loop()
#         asyncio.run_coroutine_threadsafe(self.broadcast_async(message), loop)

# websocket_manager = WebSocketManager()