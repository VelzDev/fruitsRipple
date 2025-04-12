import requests
import os
from bs4 import BeautifulSoup

def download_images(query, folder, max_images=20):
    os.makedirs(folder, exist_ok=True)
    url = f"https://www.bing.com/images/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    images = soup.find_all("img", class_="mimg")[:max_images]
    for i, img in enumerate(images):
        try:
            img_url = img["src"]
            img_data = requests.get(img_url).content
            with open(f"{folder}/{i}.jpg", "wb") as f:
                f.write(img_data)
                print("Скачано")
        except Exception as e:
            print(f"Ошибка {img_url}: {e}")

download_images("unripe apple", "dataset/unripe")
download_images("ripe apple", "dataset/ripe")
download_images("overripe apple", "dataset/overripe")