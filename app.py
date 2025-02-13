from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import base64

# Cargar el modelo YOLO entrenado
model = YOLO("best.pt")  # Reemplaza con el nombre de tu modelo

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renderizar la página principal."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Recibe una imagen, la procesa y devuelve la predicción de YOLO con bounding boxes."""
    try:
        # Leer la imagen en memoria
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Realizar la predicción con YOLO
        results = model(img)

        # Dibujar las predicciones en la imagen
        for result in results:
            for box in result.boxes:
                # Coordenadas del bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convertir imagen a base64 para enviar al frontend
        _, buffer = cv2.imencode(".jpg", img)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(content={"image": f"data:image/jpeg;base64,{encoded_image}"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
