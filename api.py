from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import os

from sam import generate_split_images
from gemini_extract_vertical_V import extract_vertical_V_for_folder

app = FastAPI()

raw_origins = os.getenv("CORS_ORIGINS", "")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

# fallback cho local nếu env chưa set
if not origins:
    origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_pdf(
    background_tasks: BackgroundTasks,    # <--- THÊM
    file: UploadFile = File(...),
    pages: str = Form(...),
):

    # 1. Parse pages
    try:
        page_list = [int(p.strip()) for p in pages.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="pages phải là list số, ví dụ: 1,2,3")

    if not page_list:
        raise HTTPException(status_code=400, detail="Phải chọn ít nhất 1 trang")

    if len(page_list) > 5:
        raise HTTPException(status_code=400, detail="Tối đa chỉ được chọn 5 trang")

    # 2. Tạo thư mục tạm
    session_id = str(uuid.uuid4())
    work_dir = Path("work") / session_id
    pdf_path = work_dir / "input.pdf"
    split_dir = work_dir / "output_split"
    out_dir = work_dir / "results"

    work_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 3. Lưu file PDF
        with pdf_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # 4. Xử lý PDF
        generate_split_images(
            pdf_path=str(pdf_path),
            split_dir=str(split_dir),
            margin=10,
            pages=page_list,
        )

        # 5. Extract
        extract_vertical_V_for_folder(
            image_dir=str(split_dir),
            pattern="*.png",
            out_dir=str(out_dir),
            md_filename="combined_vertical_V.md",
            csv_filename="combined_vertical_V.csv",
        )

        # 6. Lấy file output
        csv_path = out_dir / "combined_vertical_V.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=500, detail="Không tìm thấy file CSV kết quả")

        # 7. Đăng ký xoá folder SAU KHI RESPONSE hoàn tất
        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)

        # 8. Trả file cho FE
        return FileResponse(
            path=csv_path,
            filename="vertical_V.csv",
            media_type="text/csv",
        )

    except Exception as e:
        # Nếu lỗi, vẫn xoá folder
        shutil.rmtree(work_dir, ignore_errors=True)
        raise e
