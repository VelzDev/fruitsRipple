import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Загружаем модель
model = tf.keras.models.load_model("apple_classifier.h5")

# Классы с русскими названиями
class_labels = ["Незрелое", "Зрелое", "Перезрелое"]

# Путь к картинке
img_path = "dataset/validation/ripe/aug_apple_2.jpg"  # Подставь свою картинку

# Загружаем и подготавливаем картинку
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Делаем предсказание
predictions = model.predict(img_array)
predicted_class = class_labels[np.argmax(predictions)]
confidence = np.max(predictions) * 100  # Вероятность в %

# Рекомендация по сбору
recommendations = {
    "Незрелое": "Ожидать сбора.",
    "Зрелое": "Рекомендуется собирать.",
    "Перезрелое": "Срочно собирать!",
}

print(f"🍏 Предсказание: {predicted_class} ({confidence:.2f}%)")
print(f"📢 Рекомендация: {recommendations[predicted_class]}")