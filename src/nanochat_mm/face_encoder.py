"""Production ArcFace encoder (insightface buffalo_l w600k_r50), matching the
preprocessing used to build the cached arcface_*.npz embeddings: BGR, 112x112,
(x-127.5)/128, CHW, L2-normalised 512-d output."""
import numpy as np, cv2, onnxruntime as ort
from pathlib import Path
_MODEL = "/home/ubuntu/.insightface/models/buffalo_l/w600k_r50.onnx"

class ArcFaceEncoder:
    def __init__(self, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        try:
            self.sess = ort.InferenceSession(_MODEL, providers=list(providers))
        except Exception:
            self.sess = ort.InferenceSession(_MODEL, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def encode_pil(self, pil_rgb):
        a = np.array(pil_rgb.convert("RGB"))[..., ::-1]          # RGB->BGR
        a = cv2.resize(a, (112, 112), interpolation=cv2.INTER_LINEAR)
        x = ((a.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        e = self.sess.run(None, {self.inp: x})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


_DET = "/home/ubuntu/.insightface/models/buffalo_l/det_10g.onnx"

class FaceDetector:
    """RetinaFace detect + 5-landmark align -> aligned 112x112 BGR crop, ready for
    ArcFace. This is the realistic path: a VLM gives a rough region, we re-detect and
    align the actual face inside it (handles box imprecision and un-aligned scenes)."""
    def __init__(self, gpu=True):
        from insightface.model_zoo import model_zoo
        provs = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.det = model_zoo.get_model(_DET, providers=provs)
        use_cuda = "CUDAExecutionProvider" in self.det.session.get_providers()
        self.det.prepare(ctx_id=0 if use_cuda else -1, input_size=(640, 640))

    def detect_align(self, pil_rgb):
        from insightface.utils import face_align
        img = np.array(pil_rgb.convert("RGB"))[..., ::-1]            # BGR
        bboxes, kpss = self.det.detect(img, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        i = int(areas.argmax())                                      # largest face in region
        return face_align.norm_crop(img, landmark=kpss[i], image_size=112)  # BGR 112

class ArcFaceEncoderBGR(ArcFaceEncoder):
    def encode_bgr112(self, bgr112):
        x = ((bgr112.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        e = self.sess.run(None, {self.inp: x})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)
