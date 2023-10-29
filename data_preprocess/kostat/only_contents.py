import os
import re
import json
import shutil
from multiprocessing import Pool

import tqdm
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def pixel_to_point_bounding_box(pdf_height, pixel_bounding_box):
    def pixel_to_point(val):
        return val * 72 / 200  # Convert pixel to point using 200 DPI

    def adjust_y(y, height):
        return height - y  # Adjust Y-coordinate
    
    x1, y1, x2, y2 = [pixel_to_point(pixel_value) for pixel_value in pixel_bounding_box]  # Convert pixel values to point
    point_bounding_box = (x1, adjust_y(y2, pdf_height), x2, adjust_y(y1, pdf_height)) # image layout bounding box (x1, y1, x2, y2)
    return point_bounding_box

def inside(center, box):
    cx, cy = center
    bx1, by1, bx2, by2 = box
    return bx1 <= cx <= bx2 and by1 <= cy <= by2

def get_json_data(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def get_text_layouts(json_data):
    text_layouts = extract_pages(json_data['url'])
    datas = []
    for text_layout, single_page_contents in zip(text_layouts, json_data['content']):
        pdf_height = text_layout.height
        for content in single_page_contents:
            if content['source_type'] == 'contents':
                pdf_height = text_layout.height
                main_bbox = content['main_bbox'] # Get the four corners pixel values of the box (x1, y1, x2, y2)
                pixel_bounding_box = [main_bbox['x_1'], main_bbox['y_1'], main_bbox['x_2'], main_bbox['y_2']]
                bounding_box = pixel_to_point_bounding_box(pdf_height, pixel_bounding_box)
                texts_inside = []
                for element in text_layout:
                    if isinstance(element, LTTextContainer):
                        e_x1, e_y1, e_x2, e_y2 = element.bbox
                        e_center = ((e_x1 + e_x2) / 2, (e_y1 + e_y2) / 2)
                        
                        if inside(e_center, bounding_box):
                            texts_inside.append((e_y2, e_x1, element.get_text()))  # Y, X, and text

                if len(texts_inside) == 0:
                    continue
                texts_inside.sort(key=lambda x: (-x[0], x[1]))

                description = ''
                for _, _, text in texts_inside:
                    description += text.replace('\n', ' ')
                description = re.sub(' +', ' ', description).strip()
                datas.append({'image_path':content['crop_image_path'], 'text': description})     
    return datas

def sanitize_filename(filename):
    # Unix/Linux + Windows에 대한 허용되지 않는 문자들
    not_allowed = r'[\<>:\"/|?*]'
    sanitized = re.sub(not_allowed, "_", filename)
    return sanitized

def save_contents_data(datas, image_save_root_path, json_save_root_path):
    for data in datas:
        image_path = data['image_path']
        
        if not os.path.exists(image_path):
            continue

        text = data['text']
        if (len(text) > 400) or (len(text) < 10):
            continue
        image_name = '_'.join(image_path.split('/')[-2:])
        image_name = sanitize_filename(image_name)

        image_save_path = os.path.join(image_save_root_path, image_name)
        json_save_path = os.path.join(json_save_root_path, image_name.replace('.png', '.json'))

        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump({'image_path': image_save_path, 'text': text}, f, indent=4, ensure_ascii=False)

        shutil.copy(image_path, image_save_path)

def process_json_file(json_file):
    try:
        json_data = get_json_data(os.path.join(json_root_path, json_file))
        datas = get_text_layouts(json_data)
        save_contents_data(datas, image_save_root_path, json_save_root_path)
    except:
        print('Error: ', json_file)

if __name__ == '__main__':
    json_root_path = '/root/inference/train_data/kostat/json'
    image_save_root_path = '/root/inference/train_data/kostat/contents/image/'
    json_save_root_path = '/root/inference/train_data/kostat/contents/json/'
    
    os.makedirs(image_save_root_path, exist_ok=True)
    os.makedirs(json_save_root_path, exist_ok=True)

    json_file_list = os.listdir(json_root_path)
    # 중복제거
    current_json_file_list = set(list(file.split('.pdf')[0]+'.json' for file in os.listdir(json_save_root_path)))
    json_file_list = list(set(json_file_list) - current_json_file_list)
    
    # 멀티프로세싱을 위한 Pool 생성
    # cpu_count()는 사용 가능한 CPU 코어 수를 반환합니다.
    # 해당 값을 사용하면 시스템의 모든 코어를 사용하게 됩니다.
    with Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(process_json_file, json_file_list), total=len(json_file_list)))