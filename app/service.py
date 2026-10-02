import os
import uuid

import aiofiles
from fastapi import UploadFile, HTTPException

from app.config import get_settings
from app.logger import log

settings = get_settings()


async def upload_file(file: UploadFile, document_id: uuid.UUID) -> str:
    if not file.filename:
        log.warning(f"Upload rejected: No filename provided for document {document_id}")
        raise HTTPException(
            status_code=400, detail="File must have a filename"
        )

    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        original_name, extension = os.path.splitext(file.filename)
        safe_basename = os.path.basename(original_name)
        file_path = os.path.join(
            settings.UPLOAD_DIR, f"{safe_basename}_{document_id}{extension}"
        )

        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await file.read(1024 * 1024):
                await out_file.write(content)

        log.info(f"Successfully saved file to {file_path}")
        return file_path

    except OSError as e:
        log.error(f"Failed to save file {document_id} to disk: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to save file to disk"
        )
