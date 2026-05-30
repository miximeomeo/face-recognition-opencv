"""
train.py — Huấn luyện LBPH Face Recognizer
===========================================
Chạy:  python train.py
Đọc toàn bộ ảnh trong dataset/, trích xuất label từ tên file,
rồi huấn luyện và lưu mô hình vào trainer/trainer.yml
"""

import cv2
import numpy as np
import os
from PIL import Image

# ── Cấu hình ────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
TRAINER_DIR  = "trainer"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Tham số LBPH (có thể điều chỉnh)
LBPH_RADIUS    = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X    = 8
LBPH_GRID_Y    = 8
# ────────────────────────────────────────────────────────────────────────────


def get_images_and_labels(dataset_path: str):
    """
    Quét dataset_path, đọc từng ảnh grayscale và trích xuất User ID
    từ tên file theo định dạng  User.<id>.<sample>.jpg

    Trả về: (face_samples, ids)
    """
    image_paths = [
        os.path.join(dataset_path, f)
        for f in sorted(os.listdir(dataset_path))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh nào trong '{dataset_path}'.\n"
            "Hãy chạy  python collect.py  trước."
        )

    face_samples: list[np.ndarray] = []
    ids: list[int] = []

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    for image_path in image_paths:
        filename = os.path.basename(image_path)

        # Trích xuất ID từ tên file "User.<id>.<sample>.jpg"
        try:
            user_id = int(filename.split(".")[1])
        except (IndexError, ValueError):
            print(f"  ⚠️  Bỏ qua file không đúng định dạng: {filename}")
            continue

        # Đọc ảnh grayscale
        pil_img  = Image.open(image_path).convert("L")
        img_arr  = np.array(pil_img, dtype=np.uint8)

        # Nếu ảnh đã là face crop (collect.py lưu trực tiếp), dùng luôn.
        # Nếu là ảnh đầy đủ (toàn cảnh), chạy detect để lấy ROI.
        if img_arr.shape[0] < 200 and img_arr.shape[1] < 200:
            # Giả sử đây là face crop
            face_samples.append(img_arr)
            ids.append(user_id)
        else:
            faces = face_cascade.detectMultiScale(img_arr, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_samples.append(img_arr[y:y + h, x:x + w])
                ids.append(user_id)

        print(f"  ✔  {filename}  →  User ID {user_id}")

    return face_samples, ids


def train():
    os.makedirs(TRAINER_DIR, exist_ok=True)

    print("=" * 50)
    print("  LBPH Face Recognizer — Huấn luyện")
    print("=" * 50)
    print(f"\n📂 Đang đọc ảnh từ '{DATASET_DIR}/' ...\n")

    faces, ids = get_images_and_labels(DATASET_DIR)

    unique_users = sorted(set(ids))
    print(f"\n📊 Tổng số mẫu : {len(faces)}")
    print(f"👤 Số người dùng: {len(unique_users)}  (IDs: {unique_users})")

    print("\n⚙️  Đang huấn luyện LBPH ...", end=" ", flush=True)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y,
    )
    recognizer.train(faces, np.array(ids, dtype=np.int32))
    recognizer.write(TRAINER_FILE)

    print("xong!")
    print(f"\n✅ Mô hình đã lưu tại: '{TRAINER_FILE}'")
    print("Chạy  python detect.py  để nhận diện khuôn mặt.\n")


if __name__ == "__main__":
    train()
