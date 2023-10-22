import os
import json
import random

import tqdm
from PIL import Image, ImageDraw

def process_json_and_image(json_path, img_path, target_directory, min_char_num=20, max_char_num=60):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    annotations = data["text"]["word"]
    
    chunks = []
    current_chunk = []
    current_chars = 0
    
    split_chars_num = random.randint(min_char_num, max_char_num)
    for ann in annotations:
        text = ann["value"]
        if current_chars + len(text) <= split_chars_num:
            current_chunk.append(ann)
            current_chars += len(text)
        else:
            chunks.append(current_chunk)
            current_chunk = [ann]
            current_chars = len(text)
            split_chars_num = random.randint(min_char_num, max_char_num)

    if current_chunk:
        chunks.append(current_chunk)

    for idx, chunk in enumerate(chunks):        
        # Modify image
        original_img = Image.open(img_path)
        
        # Create a new blank image
        new_img = Image.new("RGB", (original_img.width, original_img.height), color="white")
        
        for ann in chunk:
            bbox = ann["wordbox"]
            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[2], bbox[3])
            cropped = original_img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
            new_img.paste(cropped, top_left)
        
        img_dir, img_file = os.path.split(img_path)
        new_img_filename = f"{os.path.splitext(img_file)[0]}_chunk_{idx}.jpg"
        new_img_path = os.path.join(target_directory, 'jpg', new_img_filename)
        new_img.save(new_img_path)

        new_json = {
            "text": " ".join([ann["value"] for ann in chunk])
        }
        json_dir, json_file = os.path.split(json_path)
        new_json_filename = f"{os.path.splitext(json_file)[0]}_chunk_{idx}.json"
        new_json_path = os.path.join(target_directory, 'json', new_json_filename)

        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_json, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 시작 디렉토리
    base_directory = "/root/다양한 형태의 한글 문자 OCR/Training"  # 변경해주세요
    target_directory = "/root/다양한 형태의 한글 문자 OCR/Chunk_Training"  # 변경해주세요
    min_char_num, max_char_num = 120, 180  # 변경해주세요

    # 결과를 저장할 폴더 만들기
    os.makedirs(os.path.join(target_directory, "json"), exist_ok=True)
    os.makedirs(os.path.join(target_directory, "jpg"), exist_ok=True)

    # JSON 및 JPG 파일을 찾아 처리
    exists_files = os.listdir(os.path.join(target_directory, "json"))
    exists_files = [os.path.splitext(filename)[0][:-8] for filename in exists_files]
    print(exists_files[:2])

    for root, dirs, files in os.walk(base_directory):
        for filename in tqdm.tqdm(files):
            if filename.endswith(".json"):
                category_path = os.path.relpath(root, base_directory).split(os.path.sep)[1:]
                corresponding_jpg_path = os.path.join(base_directory, "jpg", *category_path, filename.replace(".json", ".jpg"))
                if os.path.exists(corresponding_jpg_path) and os.path.splitext(filename)[0] not in exists_files:
                    process_json_and_image(os.path.join(root, filename), corresponding_jpg_path, target_directory, min_char_num, max_char_num)
