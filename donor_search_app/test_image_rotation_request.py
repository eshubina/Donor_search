import requests

image_path = "/app/images/t4.png"


response = requests.post("http://localhost:8000/process/", params={"image_path": image_path})

# Проверяем, что запрос успешный
if response.status_code == 200:
    # Получаем обработанное изображение
    from PIL import Image
    from io import BytesIO

    # Конвертируем байты обратно в изображение и отображаем его
    img = Image.open(BytesIO(response.content))
    img.show()  # Открывает изображение в приложении по умолчанию
else:
    print(f"Ошибка: {response.status_code} - {response.text}")
