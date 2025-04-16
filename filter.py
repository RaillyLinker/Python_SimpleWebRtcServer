import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

overlay_img = cv2.imread('overlay.png', cv2.IMREAD_UNCHANGED)  # PNG with alpha

def overlay_image(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))

    if overlay_resized.shape[2] == 4:
        overlay_rgb = overlay_resized[:, :, :3]
        mask = overlay_resized[:, :, 3] / 255.0
    else:
        overlay_rgb = overlay_resized
        mask = np.ones((h, w))

    roi = background[y:y+h, x:x+w]

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask) + overlay_rgb[:, :, c] * mask

    background[y:y+h, x:x+w] = roi
    return background

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        # 얼굴 영역을 살짝 키워서 일렁임 완화 + 잘 덮이게
        scale = 1.5  # 1.2 ~ 2.0 조절 가능
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_x = x - (new_w - w) // 2
        new_y = y - (new_h - h) // 2

        # 이미지 경계 벗어나지 않도록 클리핑
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        if new_x + new_w > frame.shape[1]: new_w = frame.shape[1] - new_x
        if new_y + new_h > frame.shape[0]: new_h = frame.shape[0] - new_y

        frame = overlay_image(frame, overlay_img, new_x, new_y, new_w, new_h)

    return frame
