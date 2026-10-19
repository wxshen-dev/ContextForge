import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.clients.minio_utils import get_minio_client
from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)

app = FastAPI(
    title="ContextForge Import Service",
    description="Upload PDF and Markdown documents to the knowledge base.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/import.html")
async def get_import_page() -> RedirectResponse:
    return RedirectResponse(url="http://127.0.0.1:8001/import", status_code=308)


def run_graph_task(task_id: str, local_dir: str, local_file_path: str) -> None:
    update_task_status(task_id, "processing")
    state = get_default_state()
    state.update(
        task_id=task_id,
        local_dir=local_dir,
        local_file_path=local_file_path,
    )
    try:
        for event in kb_import_app.stream(state):
            for node_name in event:
                add_done_task(task_id, node_name)
        update_task_status(task_id, "completed")
        logger.info("Import completed for task {}", task_id)
    except Exception:
        update_task_status(task_id, "failed")
        logger.exception("Import failed for task {}", task_id)


@app.post("/upload", summary="Upload knowledge base files")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> dict:
    task_ids: list[str] = []
    date = datetime.now().strftime("%Y%m%d")
    output_root = PROJECT_ROOT / "output" / date

    for upload in files:
        filename = Path(upload.filename or "").name
        if not filename or Path(filename).suffix.lower() not in {".pdf", ".md"}:
            raise HTTPException(status_code=400, detail="Only PDF and Markdown files are supported")

        task_id = str(uuid4())
        task_ids.append(task_id)
        add_running_task(task_id, "upload_file")
        local_dir = output_root / task_id
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename
        with local_path.open("wb") as destination:
            shutil.copyfileobj(upload.file, destination)

        try:
            client = get_minio_client()
            client.fput_object(
                bucket_name=os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket"),
                object_name=f"{os.getenv('MINIO_PDF_DIR', 'pdf_files')}/{date}/{task_id}/{filename}",
                file_path=str(local_path),
                content_type=upload.content_type,
            )
        except Exception:
            logger.exception("MinIO upload failed for task {}; continuing locally", task_id)

        add_done_task(task_id, "upload_file")
        background_tasks.add_task(run_graph_task, task_id, str(local_dir), str(local_path))

    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(task_ids)}",
        "task_ids": task_ids,
    }


@app.get("/status/{task_id}", summary="Get import task status")
async def get_task_progress(task_id: str) -> dict:
    return {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
