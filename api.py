from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import os
import re
from typing import List, Any, Dict

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

def parse_markdown_table(md: str) -> Dict[str, Any]:
    """
    Trả về:
    {
      "headers": [...],
      "rows": {
        "Y1": [...],
        "Y2": [...],
        ...
      }
    }
    """
    import re

    lines = [l.strip() for l in md.splitlines() if l.strip()]
    table_lines = [l for l in lines if l.startswith("|") and "|" in l]

    if len(table_lines) < 2:
        raise ValueError("Không tìm thấy bảng Markdown hợp lệ.")

    header_line = table_lines[0]
    data_lines = table_lines[2:]  # skip align row

    def split_row(line: str):
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        cells = [c.strip() for c in line.split("|")]
        # remove bold **Y1**
        cells = [re.sub(r"^\*\*(.*)\*\*$", r"\1", c) for c in cells]
        return cells

    headers = split_row(header_line)[1:]  # bỏ cột "Axis"

    row_dict: Dict[str, List[Any]] = {}

    for dl in data_lines:
        raw = split_row(dl)

        row_key = raw[0]      # ví dụ "Y1"
        raw_cells = raw[1:]   # giá trị số

        converted = []
        for c in raw_cells:
            if c == "" or c.lower() == "null":
                converted.append(None)
                continue
            try:
                if "." in c:
                    converted.append(float(c))
                else:
                    converted.append(int(c))
            except ValueError:
                converted.append(c)

        row_dict[row_key] = converted

    return {
        "headers": headers,
        "rows": row_dict
    }


@app.post("/process")
async def process_pdf(
    background_tasks: BackgroundTasks,    # <--- THÊM
    file: UploadFile = File(...),
    pages: str = Form(...),
    prompt: str | None = Form(None),
):

    # 1. Parse pages
    try:
        page_list = [int(p.strip()) for p in pages.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="pages phải là list số, ví dụ: 1,2,3")

    if not page_list:
        raise HTTPException(status_code=400, detail="Phải chọn ít nhất 1 trang")


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
            prompt=prompt,
        )

        md_path = out_dir / "combined_vertical_V.md"
        if not md_path.exists():
            raise HTTPException(status_code=500, detail="Không tìm thấy file Markdown kết quả")

        md_content = md_path.read_text(encoding="utf-8")

        # Parse markdown -> headers + rows
        table_json = parse_markdown_table(md_content)

        # Xoá folder sau khi trả xong
        background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)

        # Trả JSON đúng format yêu cầu
        return table_json

    except Exception as e:
        # Nếu lỗi, vẫn xoá folder
        shutil.rmtree(work_dir, ignore_errors=True)

        # Nếu lỗi do thiếu prompt file (RuntimeError từ gemini_extract_vertical_V)
        if isinstance(e, RuntimeError):
            # Trả về 400 cho FE biết do thiếu cấu hình prompt
            raise HTTPException(status_code=400, detail=str(e))

        raise e
