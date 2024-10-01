from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import os
import argparse
import uvicorn
from model import RotationModel

# Инициализация FastAPI приложения
app = FastAPI()

# Инициализация модели
model = RotationModel("resultmobilenet_v2.pth")


# Маршрут для обработки изображений
@app.post("/process/")
async def process_image(image_path: str = Query(...)):
    try:
        # Проверяем, существует ли файл
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        # Открываем изображение
        image_file = Image.open(image_path)

        # Получаем предсказанный угол поворота от модели
        angle = model.predict_rotation(image_file)

        # Если угол поворота не 0, поворачиваем изображение
        if angle != 0:
            image_file = image_file.rotate(angle, expand=True)

        # Сохраняем изображение в памяти
        img_byte_arr = io.BytesIO()
        image_file.save(img_byte_arr, format='TIFF')
        img_byte_arr.seek(0)

        # Возврат изображения в качестве потока
        return StreamingResponse(img_byte_arr, media_type="image/tiff")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, host=args['host'], port=args['port'])
