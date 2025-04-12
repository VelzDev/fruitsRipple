import tkinter as tk
from tkinter import filedialog, Toplevel, Text, Scrollbar
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import csv

# Загрузка моделей
ripeness_model = tf.keras.models.load_model("apple_classifier.h5")
yolo_model = YOLO("yolov8n.pt")  # Кастомную модель можно впихнуть сюда

# Классы и рекомендации
class_labels = ["Незрелое", "Зрелое", "Перезрелое"]
recommendations = {
    "Незрелое": "Ожидать сбора.",
    "Зрелое": "Рекомендуется собирать.",
    "Перезрелое": "Срочно собирать!",
}

# Функция для предсказания зрелости и вычисления времени до сбора
def predict_ripeness(crop):
    try:
        crop = cv2.resize(crop, (224, 224)) / 255.0
        crop = np.expand_dims(crop, axis=0)
        preds = ripeness_model.predict(crop)
        idx = np.argmax(preds)
        confidence = np.max(preds) * 100
        label = class_labels[idx]

        # Определение времени до сбора в зависимости от уверенности предсказания
        if label=="Незрелое":
            if confidence < 60:
                wait_days = 3  # Ожидать 3 дня
            elif 60 <= confidence < 80:
                wait_days = 2  # Ожидать 2 дня
            else:
                wait_days = 1  # Собрать сразу
        else:
            wait_days = 0

        return label, confidence, wait_days
    except:
        return "Ошибка", 0, "Ошибка классификации"

# Интерфейс
class AppleRipenessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍏 YOLO + Зрелость яблок")
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()
        self.btn = tk.Button(root, text="📸 Загрузить изображение сада", command=self.load_image)
        self.btn.pack(pady=10)
        self.btn_table = tk.Button(root, text="📊 Показать таблицу", command=self.show_table)
        self.btn_table.pack(pady=10)

        self.img_cv = None
        self.photo = None
        self.boxes = []
        self.apple_data = []  # Данные о яблоках для таблицы
        self.count = 1  # Нумерация яблок

        self.canvas.bind("<Button-3>", self.show_zoom)  # Правый клик для лупы

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.img_cv = cv2.imread(path)
        results = yolo_model(self.img_cv)[0]

        img_pil = Image.fromarray(cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        self.boxes = []
        self.apple_data = []  # Очищаем старые данные о яблоках

        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            crop = self.img_cv[y1:y2, x1:x2]
            label, conf, wait_days = predict_ripeness(crop)
            color = "green" if label == "Зрелое" else "yellow" if label == "Незрелое" else "red"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Пишем только цифры побольше
            draw.text((x1 + 5, y1 - 20), f"{self.count}", fill=color, font=ImageFont.truetype("arial.ttf", 20))

            # Обновляем данные для таблицы
            self.apple_data.append({
                "№": self.count,
                "Зрелость": label,
                "Доверие": f"{conf:.1f}%",
                "Рекомендация": f"Ожидать сбора через {wait_days} дней"
            })
            self.count += 1

        img_resized = img_pil.resize((800, 600))
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.image = self.tk_img

        # Сохраняем данные в таблицу
        self.save_to_csv()

    def save_to_csv(self):
        # Сохраняем данные в CSV
        with open('apple_data.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["№", "Зрелость", "Доверие", "Рекомендация"])
            writer.writeheader()
            for row in self.apple_data:
                writer.writerow(row)
        print("Данные сохранены в apple_data.csv")

    def show_zoom(self, event):
        # Отображаем лупу
        if self.img_cv is None:
            return

        x, y = event.x, event.y
        zoom_radius = 100
        zoomed_img = self.img_cv[y - zoom_radius:y + zoom_radius, x - zoom_radius:x + zoom_radius]
        zoomed_img_pil = Image.fromarray(cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB))
        zoomed_img_resized = zoomed_img_pil.resize((zoom_radius * 2, zoom_radius * 2), Image.ANTIALIAS)

        # Удаляем предыдущую лупу, если она есть
        if hasattr(self, 'zoomed_image'):
            self.canvas.delete(self.zoomed_image)

        # Добавляем лупу на холст
        self.zoomed_image = self.canvas.create_image(x + 10, y - 10, image=ImageTk.PhotoImage(zoomed_img_resized))

    def show_table(self):
        # Создаём окно для отображения таблицы
        table_window = Toplevel(self.root)
        table_window.title("Таблица данных о яблоках")

        text_widget = Text(table_window, wrap=tk.WORD, width=100, height=20)
        text_widget.pack()

        # Заполняем таблицу данными о яблоках
        text_widget.insert(tk.END, "№  |  Зрелость  |  Доверие  |  Рекомендация\n")
        text_widget.insert(tk.END, "-"*60 + "\n")

        for apple in self.apple_data:
            text_widget.insert(tk.END, f"{apple['№']}  |  {apple['Зрелость']}  |  {apple['Доверие']}  |  {apple['Рекомендация']}\n")
        
        # Добавляем прокрутку
        scroll = Scrollbar(table_window, command=text_widget.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scroll.set)

# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = AppleRipenessApp(root)
    root.mainloop()
