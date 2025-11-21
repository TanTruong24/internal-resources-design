from pathlib import Path
from typing import List, Optional, Tuple
import io
import fitz  # PyMuPDF
import cv2
import numpy as np


try:
    from PIL import Image  # only needed for JPEG encoding
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

def read_file_as_binary(file_path):
    """
    Reads a local file and returns its binary content.

    Args:
        file_path (str): Path to the file.

    Returns:
        bytes: Binary content of the file.
    """
    with open(file_path, "rb") as f:
        binary_data = f.read()
    return binary_data


def pdf_bytes_to_images(
    pdf_bytes: bytes,
    dpi: int = 200,
    fmt: str = "png",                   # "png" or "jpg"
    save_dir: Optional[Path | str] = None,
    base_filename: Optional[str] = None,  # used for saved filenames only
    jpg_quality: int = 90
) -> List[bytes]:
    """
    Render each page of a (scanned) PDF (given as bytes) to images and return a list of image bytes.
    Optionally save images to disk as <base_filename>_p<page>.<ext> (1-based page numbering).

    Args:
        pdf_bytes: Raw bytes of the PDF.
        dpi: Output DPI (controls resolution). 72 dpi == 1.0 zoom.
        fmt: "png" (native via PyMuPDF) or "jpg".
        save_dir: If provided, images are also saved to this directory.
        base_filename: If saving, the basename to use (default: "document").
        jpg_quality: JPEG quality (1–95) if fmt="jpg".

    Returns:
        List of bytes objects, one per page, in the requested format.
    """
    fmt = fmt.lower()
    if fmt not in {"png", "jpg", "jpeg"}:
        raise ValueError("fmt must be 'png' or 'jpg'")

    if fmt in {"jpg", "jpeg"} and not _HAS_PIL:
        raise RuntimeError("JPEG output requires Pillow (pip install Pillow)")

    # Prepare output directory (optional)
    out_dir: Optional[Path] = None
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not base_filename:
            base_filename = "document"

    if not base_filename:
        base_filename = "document"

    # DPI -> zoom factor
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: List[bytes] = []

    # Open from bytes
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise RuntimeError("This PDF is encrypted and needs a password.")
        page_count = doc.page_count

        for page_index, page in enumerate(doc, start=1):
            # Render to pixmap (no alpha for cleaner JPG/PNG)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            if fmt == "png":
                img_bytes = pix.tobytes(output="png")  # PyMuPDF-native PNG
                ext = "png"
            else:
                # Convert Pixmap -> PIL Image -> JPEG bytes
                # pix.samples are RGB bytes; pix.stride is bytes per row
                mode = "RGB" if pix.n in (3, 4) else "L"
                pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=jpg_quality, optimize=True)
                img_bytes = buf.getvalue()
                ext = "jpg"

            images.append(img_bytes)

            # Optional save
            if out_dir is not None:
                filename = f"{base_filename}_p{page_index}.{ext}"
                (out_dir / filename).write_bytes(img_bytes)

    return images

# Bounding box type: (x_min, y_min, x_max, y_max)
BBox = Tuple[int, int, int, int]


def crop_image_array(img: np.ndarray, bbox: BBox, margin=0) -> np.ndarray:
    """
    Crop an image (NumPy array) using pixel coordinates.

    Args:
        img: HxWxC or HxW image (BGR if from cv2).
        bbox: (x_min, y_min, x_max, y_max) in *pixel coordinates*.

    Returns:
        Cropped image as NumPy array.
    """
    if img is None:
        raise ValueError("crop_image_array: img is None")

    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    # Clamp to image bounds
    x_min = int(max(0, min(x_min-margin, w)))
    x_max = int(max(0, min(x_max+margin, w)))
    y_min = int(max(0, min(y_min-margin, h)))
    y_max = int(max(0, min(y_max+margin, h)))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid bbox after clamping: {bbox}")

    # NumPy slices: [y_min:y_max, x_min:x_max]
    crop = img[y_min:y_max, x_min:x_max].copy()
    return crop


def crop_image_file(
    input_path: str,
    bbox: BBox,
    margin=0,
    output_path: str = None,
) -> np.ndarray:
    """
    Load an image from disk, crop by pixel bbox, optionally save.

    Args:
        input_path: path to input image.
        bbox: (x_min, y_min, x_max, y_max) in pixels.
        output_path: where to save cropped image (if not None).

    Returns:
        Cropped image as NumPy array.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    crop = crop_image_array(img, bbox, margin=margin)

    if output_path is not None:
        cv2.imwrite(output_path, crop)

    return crop

def image_array_to_png_bytes(img: np.ndarray) -> bytes:
    """
    Encode an OpenCV image (np.ndarray, BGR) as PNG bytes.
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image to PNG")
    return buf.tobytes()

def split_image_bytes_into_two(img_bytes: bytes, margin=0) -> tuple[bytes, bytes]:
    # bytes -> np.ndarray (BGR)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]

    top_bbox = (0, 0, w, h // 2)
    bottom_bbox = (0, h // 2, w, h)

    top_img = crop_image_array(img, top_bbox, margin=margin)
    bottom_img = crop_image_array(img, bottom_bbox, margin=margin)

    # np.ndarray -> PNG bytes
    top_bytes = image_array_to_png_bytes(top_img)
    bottom_bytes = image_array_to_png_bytes(bottom_img)

    return top_bytes, bottom_bytes



def generate_split_images(
    pdf_path: str,
    split_dir: str = "output_split",
    margin: int = 10,
    pages: Optional[List[int]] = None,
):
    """
    Đọc file PDF, convert sang ảnh cho các trang được chọn (pages),
    cắt mỗi trang thành top/bottom và lưu vào split_dir.

    - pdf_path: đường dẫn file PDF input
    - split_dir: thư mục output cho ảnh top/bottom
    - margin: khoảng cách margin khi cắt top/bottom
    - pages: danh sách trang 1-based, ví dụ [1, 3, 4]; None = tất cả
    """
    pdf_bytes = read_file_as_binary(pdf_path)
    imgs = pdf_bytes_to_images(pdf_bytes, pages=pages)
    print(f"[sam] Returned {len(imgs)} images as bytes.")

    split_dir_path = Path(split_dir)
    split_dir_path.mkdir(exist_ok=True, parents=True)

    for i, page_bytes in enumerate(imgs, start=1):
        top_b, bottom_b = split_image_bytes_into_two(page_bytes, margin=margin)

        top_img = cv2.imdecode(np.frombuffer(top_b, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(split_dir_path / f"page{i}_top.png"), top_img)

        bottom_img = cv2.imdecode(np.frombuffer(bottom_b, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(split_dir_path / f"page{i}_bottom.png"), bottom_img)

    print(f"[sam] Đã sinh {len(imgs)*2} ảnh top/bottom vào: {split_dir_path}")

if __name__ == "__main__":
    # Nếu bạn vẫn muốn chạy riêng sam.py
    generate_split_images()