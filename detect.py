"""
detect.py — Nhận diện khuôn mặt real-time (Haar Cascade + LBPH)
===============================================================
Chạy:  python detect.py
Nhấn 'q' để thoát.

Tuỳ chỉnh NAMES để hiển thị tên thay vì "User <id>".
"""

import cv2
import os
import time

# Cấu hình 
CASCADE_PATH    = "haarcascade_frontalface_default.xml"
TRAINER_FILE    = os.path.join("trainer", "trainer.yml")
CAMERA_INDEX    = 0
CONFIDENCE_THRESHOLD = 60   # Dưới ngưỡng này → "Unknown"
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Ánh xạ User ID → Tên hiển thị 
NAMES = {
    0: "Unknown",
    1: "Nguyen Duc Minh",
    2: "Tran Thi B",
    3: "Le Van C",
    
}
# ────────────────────────────────────────────────────────────────────────────


def get_name(user_id: int) -> str:
    return NAMES.get(user_id, f"User {user_id}")


def draw_label(frame, text: str, x: int, y: int, color: tuple):
    """Vẽ text có nền mờ để dễ đọc."""
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, 0.65, 2)
    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 4),
        (x + text_w + 4, y + baseline),
        (0, 0, 0), cv2.FILLED
    )
    cv2.putText(frame, text, (x + 2, y - 2), FONT, 0.65, color, 2)


def detect():
    # ── Kiểm tra file cần thiết ──────────────────────────────────────────────
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy '{CASCADE_PATH}'.\n"
        
        )
    if not os.path.exists(TRAINER_FILE):
        raise FileNotFoundError(
            f"Không tìm thấy '{TRAINER_FILE}'.\n"
            "Hãy chạy  python train.py  trước."
        )

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    recognizer   = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise IOError("Không mở được webcam!")

    print("🎥 Đang nhận diện khuôn mặt ... Nhấn 'q' để thoát.\n")

    fps_time   = time.time()
    fps_count  = 0
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Mất kết nối camera!")
            break

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_time    = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Cân bằng histogram → cải thiện điều kiện sáng kém
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            user_id, confidence = recognizer.predict(face_roi)

            # LBPH: confidence càng thấp → càng giống (0 = giống hoàn toàn)
            if confidence < CONFIDENCE_THRESHOLD:
                name         = get_name(user_id)
                conf_text    = f"{name}  ({100 - int(confidence)}%)"
                box_color    = (0, 255, 0)       # xanh lá = nhận ra
            else:
                conf_text    = "Unknown"
                box_color    = (0, 0, 255)       # đỏ = không nhận ra

            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Nhãn phía trên box
            draw_label(frame, conf_text, x, y, box_color)

            # Hiện confidence score nhỏ phía dưới box
            cv2.putText(
                frame,
                f"conf: {confidence:.1f}",
                (x, y + h + 18),
                FONT, 0.45, (200, 200, 200), 1
            )

        # HUD
        h_frame, w_frame = frame.shape[:2]

        # Thanh HUD trên cùng
        cv2.rectangle(frame, (0, 0), (w_frame, 38), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, "FACE RECOGNITION  |  Haar + LBPH",
                    (10, 25), FONT, 0.6, (0, 200, 255), 1)
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (w_frame - 110, 25), FONT, 0.6, (0, 200, 255), 1)

        # Thanh HUD dưới cùng
        cv2.rectangle(frame, (0, h_frame - 32), (w_frame, h_frame),
                      (20, 20, 20), cv2.FILLED)
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}   |   Threshold: {CONFIDENCE_THRESHOLD}   |   'q' to quit",
            (10, h_frame - 10), FONT, 0.5, (160, 160, 160), 1
        )

        cv2.imshow("Face Recognition — Haar + LBPH", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Đã thoát.")


if __name__ == "__main__":
    detect()
