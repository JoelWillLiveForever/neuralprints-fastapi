import hashlib
from fastapi import UploadFile

# === ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ MD5 ХЭША ФАЙЛА ===
def calculate_md5_from_file(file: UploadFile) -> str:
    md5_hash = hashlib.md5()
    # Чтение файла порциями (это важно для больших файлов)
    for chunk in iter(lambda: file.file.read(4096), b""):
        md5_hash.update(chunk)
    return md5_hash.hexdigest()

# === ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ MD5 ХЭША СТРОКИ ===
def calculate_md5_from_bytes(data: bytes) -> str:
    md5_hash = hashlib.md5()
    md5_hash.update(data)
    return md5_hash.hexdigest()