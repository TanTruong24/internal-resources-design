#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trích xuất nội lực V theo ma trận (Y1–Y6, X1–X10) từ các ảnh nội lực
bằng Gemini (free-tier).

- Input: thư mục ảnh (ví dụ: output_split/page1_top.png, page1_bottom.png, ...)
- Output: 1 file Markdown (combined_vertical_V.md) + 1 file CSV (combined_vertical_V.csv)
  trong thư mục out-dir.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

import csv

# ==========================
# CẤU HÌNH CỘT / TRỤC
# ==========================

X_COLS = [f"X{i}" for i in range(1, 11)]
AXES = [f"**Y{i}**" for i in range(1, 7)]


# ==========================
# HÀM PARSE / MERGE BẢNG MARKDOWN
# ==========================

def parse_md_table(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Parse bảng Markdown do Gemini trả về -> (header, rows)
    rows là list[dict]: {"Axis": "**Y1**", "X1": "75", ...}
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return [], []

    # Dòng đầu tiên là header: | Axis | X1 | X2 | ...
    header_cells = [c.strip() for c in lines[0].split("|")[1:-1]]

    rows: List[Dict[str, str]] = []
    # Bỏ dòng 2 (separator), đọc từ dòng thứ 3 trở đi
    for line in lines[2:]:
        parts = [c.strip() for c in line.split("|")[1:-1]]
        if not parts:
            continue
        row = dict(zip(header_cells, parts))
        rows.append(row)

    return header_cells, rows


def merge_tables_texts(md_texts: List[str]) -> Dict[str, Dict[str, str | None]]:
    """
    Nhận list các bảng Markdown (mỗi cái từ 1 ảnh),
    trả về dict combined[axis][Xn] sau khi merge.
    """
    combined: Dict[str, Dict[str, str | None]] = {
        axis: {x: None for x in X_COLS} for axis in AXES
    }

    for text in md_texts:
        _, rows = parse_md_table(text)
        for row in rows:
            axis = row.get("Axis", "")
            if axis not in combined:
                continue

            for x in X_COLS:
                raw = row.get(x, "").strip()
                # Bỏ qua ô rỗng hoặc "null"
                if raw == "" or raw.lower() == "null":
                    continue

                # Nếu chưa có giá trị, ghi vào
                if combined[axis][x] is None:
                    combined[axis][x] = raw
                # Nếu đã có giá trị khác thì giữ nguyên (có thể log nếu muốn)
    return combined


def combined_to_markdown(combined: Dict[str, Dict[str, str | None]]) -> str:
    """Từ combined dict -> 1 bảng Markdown lớn."""
    lines: List[str] = []
    header_row = "| Axis | " + " | ".join(X_COLS) + " |"
    sep_row = "| :--- | " + " | ".join([":---"] * len(X_COLS)) + " |"
    lines.append(header_row)
    lines.append(sep_row)

    for i in range(1, 7):
        axis_key = f"**Y{i}**"
        row_vals: List[str] = []
        for x in X_COLS:
            v = combined[axis_key][x]
            row_vals.append("" if v is None else str(v))
        lines.append("| " + axis_key + " | " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def combined_to_csv(combined: Dict[str, Dict[str, str | None]], csv_path: str):
    """Ghi combined ra file CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Axis"] + X_COLS)
        for i in range(1, 7):
            axis_key = f"**Y{i}**"
            axis_short = f"Y{i}"
            row: List[str] = [axis_short]
            for x in X_COLS:
                v = combined[axis_key][x]
                row.append("" if v is None else str(v))
            writer.writerow(row)


# ==========================
# PROMPT GỐC CỦA BẠN
# ==========================

def load_prompt_from_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        raise RuntimeError(f"Không đọc được file prompt: {path}")
    
DEFAULT_MODEL = "gemini-2.5-flash"  # model rẻ/nhanh, đủ dùng cho free-tier


# ==========================
# ENV & CLIENT
# ==========================

def load_env() -> str:
    """
    Load biến môi trường từ file .env (nếu có).
    Cần .env chứa: GEMINI_API_KEY=...
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Không tìm thấy GEMINI_API_KEY hoặc GOOGLE_API_KEY.\n"
            "Hãy tạo file .env với dòng:\n"
            "    GEMINI_API_KEY=YOUR_KEY_HERE"
        )
    return api_key


def make_client() -> genai.Client:
    """
    Tạo client Gemini. SDK sẽ đọc API key từ biến môi trường.
    """
    client = genai.Client()
    return client


# ==========================
# CORE GỌI GEMINI
# ==========================

def call_gemini_for_image(
    client: genai.Client,
    image_path: Path,
    prompt: str = load_prompt_from_file("prompt_vertical_V.txt"),
    model: str = DEFAULT_MODEL,
    mime_type: str = "image/png",
) -> str:
    """
    Gửi 1 ảnh + prompt sang Gemini, nhận về text (Markdown table).
    """
    image_bytes = image_path.read_bytes()

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    response = client.models.generate_content(
        model=model,
        contents=[image_part, prompt],
    )

    return response.text


# ==========================
# HÀM PUBLIC DÙNG TRONG PIPELINE
# ==========================

def extract_vertical_V_for_folder(
    image_dir: str = "output_split",
    pattern: str = "*.png",   # <<< Gộp ALL ảnh top/bottom
    out_dir: str = "gemini_results",
    model: str = DEFAULT_MODEL,
    mime_type: str = "image/png",
    md_filename: str = "combined_vertical_V.md",
    csv_filename: str = "combined_vertical_V.csv",
):
    image_dir_path = Path(image_dir)
    if not image_dir_path.is_dir():
        raise SystemExit(f"Thư mục ảnh không tồn tại: {image_dir}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    load_env()
    client = make_client()

    # 👇 GOM TẤT CẢ FILE PNG: page*_top.png + page*_bottom.png
    images = sorted(image_dir_path.glob(pattern))
    if not images:
        raise SystemExit(
            f"Không tìm thấy ảnh nào trong {image_dir} với pattern '{pattern}'"
        )

    print(f"[Gemini] Tổng số ảnh xử lý: {len(images)}")
    print(f"[Gemini] Merge tất cả vào cùng 1 bảng → {out_dir_path}")

    all_md_texts = []

    for img in images:
        print(f"  -> Đang xử lý: {img.name}")
        try:
            result_text = call_gemini_for_image(
                client=client,
                image_path=img,
                prompt=load_prompt_from_file("prompt_vertical_V.txt"),
                model=model,
                mime_type=mime_type,
            )
            print(result_text)
            all_md_texts.append(result_text)
        except Exception as e:
            print(f"     LỖI file {img.name}: {e}")

    # 🔥 Merge toàn bộ bảng MD thành một ma trận duy nhất
    combined = merge_tables_texts(all_md_texts)

    # Ghi Markdown
    md_path = out_dir_path / md_filename
    md_path.write_text(combined_to_markdown(combined), encoding="utf-8")

    # Ghi CSV
    csv_path = out_dir_path / csv_filename
    combined_to_csv(combined, str(csv_path))

    print(f"[Gemini] Đã ghi Markdown chung: {md_path}")
    print(f"[Gemini] Đã ghi CSV chung     : {csv_path}")

# ==========================
# CLI MAIN (vẫn giữ được cách chạy trực tiếp)
# ==========================

def _main_cli():
    parser = argparse.ArgumentParser(
        description="Trích xuất ma trận V (Y1–Y6, X1–X10) từ các ảnh nội lực bằng Gemini"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="output_split",
        help="Thư mục chứa ảnh (mặc định: output_split)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Pattern lọc ảnh, ví dụ: '*.png', 'page*_top.png', 'page*_bottom.png'",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="gemini_results",
        help="Thư mục lưu kết quả (1 file .md + 1 file .csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Tên model Gemini (mặc định: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mime-type",
        type=str,
        default="image/png",
        help="MIME type ảnh: image/png hoặc image/jpeg (mặc định: image/png)",
    )

    args = parser.parse_args()

    extract_vertical_V_for_folder(
        image_dir=args.image_dir,
        pattern=args.pattern,
        out_dir=args.out_dir,
        model=args.model,
        mime_type=args.mime_type,
    )


if __name__ == "__main__":
    _main_cli()
