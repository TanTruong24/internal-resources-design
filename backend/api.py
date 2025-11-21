# api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid

from sam import generate_split_images
from gemini_extract_vertical_V import extract_vertical_V_for_folder

app = FastAPI()

# Cho phép FE trên Vercel gọi (cập nhật domain thật sau)
origins = [
    "http://localhost:3000",
    "https://your-frontend.vercel.app",  # đổi thành domain Vercel của bạn
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    pages: str = Form(...),  # vd "1,3,4"
):
    # 1. Parse pages
    try:
        page_list = [
            int(p.strip())
            for p in pages.split(",")
            if p.strip() != ""
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="pages phải là list số, ví dụ: 1,2,3")

    if not page_list:
        raise HTTPException(status_code=400, detail="Phải chọn ít nhất 1 trang")

    if len(page_list) > 5:
        raise HTTPException(status_code=400, detail="Tối đa chỉ được chọn 5 trang")

    # 2. Tạo thư mục làm việc tạm cho mỗi request
    session_id = str(uuid.uuid4())
    work_dir = Path("work") / session_id
    pdf_path = work_dir / "input.pdf"
    split_dir = work_dir / "output_split"
    out_dir = work_dir / "results"

    work_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Lưu file PDF tạm
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 4. Gọi sam.generate_split_images cho đúng các trang đã chọn
    generate_split_images(
        pdf_path=str(pdf_path),
        split_dir=str(split_dir),
        margin=10,
        pages=page_list,
    )

    # 5. Gọi Gemini để trích xuất + merge kết quả
    extract_vertical_V_for_folder(
        image_dir=str(split_dir),
        pattern="*.png",             # hoặc "page*_top.png" nếu còn dùng top/bottom
        out_dir=str(out_dir),
        md_filename="combined_vertical_V.md",
        csv_filename="combined_vertical_V.csv",
    )

    # 6. Trả lại file CSV (bạn có thể thay bằng zip, hoặc trả md tương tự)
    csv_path = out_dir / "combined_vertical_V.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=500, detail="Không tìm thấy file CSV kết quả")

    return FileResponse(
        path=csv_path,
        filename="vertical_V.csv",
        media_type="text/csv",
    )
