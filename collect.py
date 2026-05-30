"""
collect.py — Thu thập ảnh khuôn mặt từ webcam
=============================================
Chạy:  python collect.py
Nhập User ID (số nguyên) khi được hỏi.
Chương trình sẽ tự động chụp 30 ảnh khuôn mặt và lưu vào dataset/
"""

import cv2
import os

# ── Cấu hình ────────────────────────────────────────────────────────────────
DATASET_DIR    = "dataset"
CASCADE_PATH   = "haarcascade_frontalface_default.xml"
NUM_SAMPLES    = 30          # số ảnh cần thu thập cho mỗi user
CAMERA_INDEX   = 0           # 0 = webcam mặc định
DISPLAY_WIDTH  = 640
DISPLAY_HEIGHT = 480
# ────────────────────────────────────────────────────────────────────────────


def collect_faces():
    os.makedirs(DATASET_DIR, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise FileNotFoundError(
            f"Không tìm thấy file cascade: '{CASCADE_PATH}'\n"
            "Tải tại: https://github.com/opencv/opencv/tree/master/data/haarcascades"
        )

    user_id = input("Nhập User ID (số nguyên, vd: 1): ").strip()
    if not user_id.isdigit():
        print("❌ User ID phải là số nguyên!")
        return
    user_id = int(user_id)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    if not cap.isOpened():
        raise IOError("Không mở được webcam. Kiểm tra CAMERA_INDEX.")

    print(f"\n📷 Đang thu thập ảnh cho User ID: {user_id}")
    print("Nhìn thẳng vào camera. Nhấn 'q' để thoát sớm.\n")

    sample_count = 0

    while sample_count < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Không đọc được frame!")
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            sample_count += 1
            face_img = gray[y:y + h, x:x + w]
            filename = os.path.join(DATASET_DIR, f"User.{user_id}.{sample_count}.jpg")
            cv2.imwrite(filename, face_img)

            # Vẽ khung + thông tin
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Sample: {sample_count}/{NUM_SAMPLES}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # HUD góc trên trái
        cv2.putText(
            frame,
            f"User {user_id}  |  Collected: {sample_count}/{NUM_SAMPLES}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2
        )

        cv2.imshow("Face Collector  —  'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("⛔ Thoát sớm theo yêu cầu.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if sample_count > 0:
        print(f"\n✅ Đã lưu {sample_count} ảnh vào '{DATASET_DIR}/' cho User ID {user_id}.")
        print("Chạy  python train.py  để huấn luyện mô hình.")
    else:
        print("\n⚠️  Không phát hiện khuôn mặt nào. Kiểm tra ánh sáng / cascade.")


if __name__ == "__main__":
    collect_faces()
