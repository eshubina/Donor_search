import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

# Параметры
IMG_SIZE = 224  # Размер изображения для модели
device = torch.device("cpu")

# Трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Класс для инициализации и работы с моделью
class RotationModel:
    def __init__(self, model_path):
        self.device = device

        # Инициализация модели MobileNetV2 с кастомными весами
        mobilenet = models.mobilenet_v2(weights=None)
        mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=mobilenet.classifier[1].in_features, out_features=4, bias=True)
        )
        self.model = mobilenet.to(self.device)

        # Загрузка обученных весов
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()  # Перевод модели в режим инференса

    def predict_rotation(self, image: Image.Image):
        # Преобразование изображения с помощью предобработки
        img_tensor = transform(image).unsqueeze(0).to(self.device)

        # Выполняем предсказание
        with torch.no_grad():
            outputs = self.model(img_tensor)

        # Получаем класс с наибольшим значением
        predicted_class = torch.argmax(outputs, 1).item()

        # Угол поворота на основе класса (0 - 0 градусов, 1 - 90, 2 - 180, 3 - 270)
        return predicted_class * 90
