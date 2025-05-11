from fastapi.middleware.cors import CORSMiddleware

# "http://localhost:5173",
origins = [
    "*",
]

def configure_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )