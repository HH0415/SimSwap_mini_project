#（以 MediaPipe 估 landmark，生成 256x256 對齊臉）
import cv2, mediapipe as mp, numpy as np


def align_face_square(img, size=256):
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    h,w = img.shape[:2]
    res = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None, None
    lm = res.multi_face_landmarks[0]
    pts = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]
    # 粗略使用雙眼與嘴的重心作相似變換
    left_eye = np.mean([pts[i] for i in [33, 133]], axis=0)
    right_eye = np.mean([pts[i] for i in [362, 263]], axis=0)
    mouth = np.mean([pts[i] for i in [13, 14]], axis=0)
    src = np.float32([left_eye, right_eye, mouth])


# 標準 256 模板點（可依需求微調）
    dst = np.float32([[88,108], [168,108], [128,170]])


    M = cv2.getAffineTransform(src, dst)
    aligned = cv2.warpAffine(img, M, (size,size))
    return aligned, pts