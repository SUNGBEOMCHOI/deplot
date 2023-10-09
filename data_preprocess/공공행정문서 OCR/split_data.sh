#!/bin/bash

# 두 폴더의 경로를 지정합니다.
jpg_folder="/root/공공행정문서 OCR/long_process_Validation/jpg"
json_folder="/root/공공행정문서 OCR/long_process_Validation/json"

# 이동할 폴더의 경로를 지정합니다.
target_jpg_folder="/root/공공행정문서 OCR/split_long_process_Validation/jpg"
target_json_folder="/root/공공행정문서 OCR/split_long_process_Validation/json"

# 대상 폴더가 없으면 생성합니다.
mkdir -p "$target_jpg_folder"
mkdir -p "$target_json_folder"

# jpg 폴더에서 파일 목록을 가져와서 10%의 파일을 선택합니다.
num_files=$(ls -1q "$jpg_folder" | wc -l)
num_moving_files=$((num_files / 10))

# 10%의 jpg 파일을 대상 폴더로 이동합니다.
ls "$jpg_folder" | head -n "$num_moving_files" | while read -r file; do
    mv "$jpg_folder/$file" "$target_jpg_folder"
    # 동일한 이름의 json 파일도 이동합니다.
    base_name=$(basename "$file" .jpg)
    mv "$json_folder/$base_name.json" "$target_json_folder"
done
