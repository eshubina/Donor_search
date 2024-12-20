# DonorSearch

### Описание и цель проекта
DonorSearch: Deep Learning для работы с документами (CV)

Заказчик - DonorSearch -  занимается развитием донорства в стране. Для этого есть платформа DonorSearch.org - где для доноров доступны бонусная программа, игрофикация пути донора и многое другое. Важной является проверка честности доноров и корректности внесенных донаций. Подтверждение производится по справке установленной формы (№405), такую справку донор получает в центре крови.  Далее загружает как картинку или pdf в личный кабинет. 

Необходимо определить угол загружаемого изображения перед запуском сервиса OCR. \
Создание микросервиса для последующей интеграции в продукт заказчика.

### Вывод
Были обучены 4 модели: ResNet50, Mobilenet V2, DenseNet121 и VGG19. Коэффициент скорости обучения 0.001, на 10 эпохах.\
Лучшей определена Mobilenet V2 с accuracy на тестовой выборке - 0.95

#### Микросервис для определения угла фотографий медицинских справок

Этот [проект](https://github.com/eshubina/Donor_search/tree/main/donor_search_app) представляет собой веб-сервис на основе FastAPI для поворота справок, исходя из предсказаний обученной модели. Модель определяет ориентацию изображения, чтобы сервис OCR смог корректно обработать справку.

##### Запуск приложения

**1. Скачайте Docker-образ с Docker Hub:**

```bash
docker pull yarboxes/donor_search_script
```

**2. Для запуска контейнера используйте следующую команду:**

```bash
docker run -p 8000:8000 -v /path/to/your/images:/app/images --name image_rotation yarboxes/donor_search_script
```

- Замените /path/to/your/images на путь к папке с изображениями на вашем компьютере
- Приложение будет доступно по адресу http://localhost:8000

**3. Используйте следующий [скрипт](https://github.com/eshubina/Donor_search/blob/main/donor_search_app/test_image_rotation_request.py) для отправки запроса на сервер и получения обработанного изображения.**

- Замените /app/images/your_image_file.png на имя файла изображения, который находится в папке с изображениями на вашем компьютере
- Скрипт отправляет POST-запрос на сервер FastAPI и получает обработанное изображение

