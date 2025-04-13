import albumentations as A
import cv2
import os

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5)
])

input_dir = "dataset/banana/train/unripe/"
output_dir = "dataset/banana/train/unripe/"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    augmented = transform(image=img)["image"]
    cv2.imwrite(os.path.join(output_dir, f"aug_{img_name}"), augmented)