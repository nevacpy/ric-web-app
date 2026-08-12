import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="香港塑料瓶 RIC 識別系統 API")

# 允許跨域請求 (React Native 開發必備)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入兩個專用模型：RIC1 (PET) 與 RIC5 (PP)
MODELS = {
    "ric1-v5": YOLO("ric1-v5.pt"),
    "ric5-v1": YOLO("ric5-v1.pt"),
}


# 封裝 OpenCV 預處理算法
def preprocess_image(image_bytes):
    # 將上傳的二進制流轉為 OpenCV 矩陣
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 應用 CLAHE 增強對比
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Canny 邊緣檢測
    edges = cv2.Canny(enhanced, 40, 120)

    # 關鍵修正：轉回 3 通道以符合 YOLO 輸入格式
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_3ch


def extract_model_result(result, model_name: str) -> dict | None:
    """Return the highest-confidence detection for a model, or None if empty."""
    best = None
    best_conf = -1.0

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf <= best_conf:
            continue

        best_conf = conf
        cls_id = int(box.cls[0])
        best = {
            "class": result.names[cls_id],
            "confidence": round(conf, 2),
            "bbox": [int(x) for x in box.xyxy[0].tolist()],
            "model": model_name,
        }

    return best


# 建立 Post 接口供 React Native 調用
@app.post("/predict")
async def predict_bottle(file: UploadFile = File(...)):
    try:
        # 讀取前端傳來的圖片檔案
        contents = await file.read()

        # 執行預處理
        processed_img = preprocess_image(contents)

        # 每個模型各回傳一筆結果；無檢測則為 null
        detections: dict[str, dict | None] = {}
        total_inference_ms = 0.0

        for model_name, model in MODELS.items():
            results = model.predict(source=processed_img, imgsz=640, conf=0.25)
            result = results[0]
            total_inference_ms += float(result.speed.get("inference", 0.0))
            detections[model_name] = extract_model_result(result, model_name)

        # 從有檢測的模型中選最高置信度作為建議結果
        found = [d for d in detections.values() if d is not None]
        found.sort(key=lambda d: d["confidence"], reverse=True)
        suggestion_result = found[0] if found else None

        return {
            "success": True,
            "inference_speed_ms": total_inference_ms,
            "suggestion_result": suggestion_result,
            "detections": detections,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# 啟動本地伺服器
if __name__ == "__main__":
    import uvicorn

    # 監聽 0.0.0.0 確保局域網內的手機可以訪問
    uvicorn.run(app, host="0.0.0.0", port=8000)
