#（ONNX 版 ArcFace 嵌入器）
import onnxruntime as ort
import numpy as np, cv2


class ArcFaceONNX:
    def __init__(self, onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider']):
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name


def _pre(self, img):
    # img: RGB tensor (C,H,W) or np, here expect np(H,W,3)
    img = cv2.resize(img, (112,112))
    img = img[..., ::-1] # RGB->BGR if需要視模型
    img = (img - 127.5)/128.0
    img = img.transpose(2,0,1)[None].astype(np.float32)
    return img


def __call__(self, img_np):
    x = self._pre(img_np)
    feat = self.sess.run(None, {self.input_name:x})[0]
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True)+1e-6)
    return feat


@staticmethod
def cosine(a, b):
    return float(np.dot(a, b.T))