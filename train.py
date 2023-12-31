import os
import re
import json
import difflib

import numpy as np
import pandas as pd
import tqdm
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup, AdafactorSchedule

class CombinedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find the dataset which contains the item at the current index
        dataset_index = next(i for i, cumulative_length in enumerate(self.cumulative_lengths) if idx < cumulative_length)
        
        # If not the first dataset, adjust the index
        if dataset_index > 0:
            idx -= self.cumulative_lengths[dataset_index - 1]
        
        return self.datasets[dataset_index][idx]

class ChartToTableDataset(Dataset):
    def __init__(self, image_dir, txt_dir):
        self.image_dir = image_dir
        self.txt_dir = txt_dir
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.txt_filenames = sorted(os.listdir(self.txt_dir))
        assert len(self.image_filenames) == len(self.txt_filenames), "Number of images and TXT files do not match!"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        txt_path = os.path.join(self.txt_dir, self.txt_filenames[idx])
        image = Image.open(image_path).convert("RGB")
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            # Don't replace commas in the header
            line = line.replace('\n', ' <0x0A> ')
            line = re.sub(r'\s+\|', ' |', line)
            line = re.sub(r'\|\s+', '| ', line)
            lines[i] = line

        text = ''.join(lines)

        return {
            "image": image,
            "text": text
        }


class ChartQADataset(Dataset):
    def __init__(self, image_dir, csv_dir):
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.csv_filenames = sorted(os.listdir(self.csv_dir))
        assert len(self.image_filenames) == len(self.csv_filenames), "Number of images and CSV files do not match!"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        csv_path = os.path.join(self.csv_dir, self.csv_filenames[idx])
        image = Image.open(image_path).convert("RGB")

        # Load csv content and replace commas with pipes
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # Don't replace commas in the header
                line_parts = line.split(',')
                line = ' | '.join(line_parts)
                line = line.replace('\n', ' <0x0A> ')
                lines[i] = line

            text = ''.join(lines)
            text = text.replace('"', '')

        return {
            "image": image,
            "text": text
        }

class PublicOCRDataset(Dataset):
    def __init__(self, image_dir, json_dir):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.json_filenames = sorted(os.listdir(self.json_dir))
        assert len(self.image_filenames) == len(self.json_filenames), "Number of images and JSON files do not match!"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")

        # Load csv content and replace commas with pipes
        json_path = os.path.join(self.json_dir, self.json_filenames[idx])
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data['text'].strip()

        return {
            "image": image,
            "text": text
        }

class PartitionKostatDataset(Dataset):
    def __init__(self, image_dir, bbox_dir, part_dir):
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.part_dir = part_dir
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.bbox_filenames = sorted(os.listdir(self.bbox_dir))
        self.part_filenames = sorted(os.listdir(self.part_dir))
        assert len(self.image_filenames) == len(self.bbox_filenames) == len(self.part_filenames) , "Number of images, bbox and part files do not match!"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")
        bbox_path = os.path.join(self.bbox_dir, self.bbox_filenames[idx])
        part_path = os.path.join(self.part_dir, self.part_filenames[idx])
        bbox_text = ''
        with open(bbox_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            cnt = 1
            for line in lines:
                line = line.split()
                class_num, bbox = line[0], line[1:]
                if class_num == 0:
                    continue
                # Don't replace commas in the header
                bbox_text += f'{cnt} {" ".join(bbox).strip()} <0x0A> '
                cnt += 1
        
        part_text = ''
        with open(part_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                # Don't replace commas in the header
                part_text += line.replace('\n', ' <0x0A> ')        

        return {
            "image": image,
            "bbox_text": bbox_text,
            "text": part_text
        }

# Now, let's integrate this with the ImageCaptioningDataset you've provided:

MAX_PATCHES = 2048

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_dir, csv_dir, processor):
        self.dataset = ChartQADataset(image_dir, csv_dir)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text="Generate underlying data table of the figure below:", return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

class MixedChartToTableDataset(Dataset):
    def __init__(self, image_dir, txt_dir, processor):
        self.dataset = ChartToTableDataset(image_dir, txt_dir)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text="Generate underlying data table of the figure below:", return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

class OCRDataset(Dataset):
    def __init__(self, image_dir, json_dir, processor):
        self.dataset = PublicOCRDataset(image_dir, json_dir)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text="Optical character recognition below:", return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)
        
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

class PartitionDataset(Dataset):
    def __init__(self, image_dir, bbox_dir, part_dir, processor):
        self.dataset = PartitionKostatDataset(image_dir, bbox_dir, part_dir)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        bbox_text = item['bbox_text']
        encoding = self.processor(images=item["image"], text=f"Generate a hierachical relation of the below: bounding box {bbox_text}, figure:", return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)
        # encoding = self.processor(images=item["image"], text=f"Generate a hierachical relation of the below: :", return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding

def collator(batch):
    new_batch = {"flattened_patches":[], "attention_mask":[]}
    texts = [item["text"] for item in batch]

    text_inputs = processor.tokenizer(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=400, truncation=True)
    new_batch['raw_labels'] = texts
    new_batch["labels"] = text_inputs.input_ids

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch



from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps = 8)

def train(model, epochs=50, train_dataloader=None):
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=0.0001, weight_decay=1e-05)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    model.train()

    for epoch in range(epochs):
        print("Epoch:", epoch)
        total_loss = 0
        for idx, batch in enumerate(tqdm.tqdm(train_dataloader)):
            with accelerator.accumulate(model):
                labels = batch.pop("labels")
                flattened_patches = batch.pop("flattened_patches")
                attention_mask = batch.pop("attention_mask")

                if ((epoch + 1) % 10 == 0) and ((idx+1) % 100 == 0):
                    model.eval()

                    predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)
                    print("---------Predictions---------\n", processor.batch_decode(predictions, skip_special_tokens=True))
                    print()
                    print("---------labels---------\n", batch.pop("raw_labels")[0])

                    model.train()
                
                outputs = model(flattened_patches=flattened_patches,
                                attention_mask=attention_mask,
                                labels=labels)

                loss = outputs.loss
                total_loss += loss.item()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # scheduler.step()
                # if ((epoch + 1) % 20 == 0) and ((idx+1) % 100 == 0):
                
        if (epoch + 10) % 1 == 0:
            model.save_pretrained(f'{epoch+1}_model')
        with open('kostat_graph_loss.txt', 'a') as f:
            f.write("Loss:" + str(total_loss/len(train_dataloader)) + "\n")
    

def relative_distance(p, t):
    if t == 0:
        # if both are zero, they are the same
        return 0 if p == 0 else 1
    return min(1, np.abs(p - t) / np.abs(t))

def compute_rnss(predicted_table, target_table):
    # Extract numbers from tables
    predicted_numbers = [float(item) for item in predicted_table.split() if item.replace('.', '', 1).isdigit()]
    target_numbers = [float(item) for item in target_table.split() if item.replace('.', '', 1).isdigit()]

    N, M = len(predicted_numbers), len(target_numbers)

    # If there are no numbers, return a default value (e.g., 0) or handle appropriately
    if N == 0 or M == 0:
        return 0

    # Compute pairwise set of relative distances
    distance_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            distance_matrix[i][j] = relative_distance(predicted_numbers[i], target_numbers[j])

    # Find minimal cost matching
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Calculate RNSS score
    rnss = 1 - distance_matrix[row_ind, col_ind].sum() / max(N, M)
    return rnss

def compute_similarity(predicted_string, target_string):
    # Create a SequenceMatcher
    matcher = difflib.SequenceMatcher(None, predicted_string, target_string)
    
    # Get the similarity ratio
    return matcher.ratio()

def compute_accuracy(predictions, labels):
    total_score = 0.0
    
    for pred, label in zip(predictions, labels):
        # rnss_score = compute_rnss(pred, label)
        # total_score += rnss_score
        similarity_score = compute_similarity(pred, label)
        total_score += similarity_score
        
    # Average the scores
    return total_score / len(predictions)

def test(model, dataloader):    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    for idx, batch in enumerate(tqdm.tqdm(dataloader)):
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=512)
        decoded_predictions = processor.batch_decode(predictions, skip_special_tokens=True)
        prediction = decoded_predictions[0]
        label = batch.pop("raw_labels")[0]
        all_predictions.append(prediction)
        all_labels.append(label)
        
        print("---------Predictions---------\n", prediction)
        print('\n')
        print("---------labels---------\n", label)

    accuracy = compute_accuracy(all_predictions, all_labels)
    print(f"RNSS: {accuracy :.2f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # processor = Pix2StructProcessor.from_pretrained('google/deplot')
    # model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(device)
    processor = Pix2StructProcessor.from_pretrained('./final_model')
    # model = Pix2StructForConditionalGeneration.from_pretrained('./final_model').to(device)
    # model = Pix2StructForConditionalGeneration.from_pretrained('./chartqa_plotqa_5').to(device)
    # model = Pix2StructForConditionalGeneration.from_pretrained('./kostat_ocr_model').to(device)
    model = Pix2StructForConditionalGeneration.from_pretrained('./24_model').to(device)
    processor.image_processor.is_vqa = True

    # for ChartQA
    #============================#
    # train_image_dir="/root/ChartQA/ChartQA Dataset/translate_train/png"
    # train_csv_dir="/root/ChartQA/ChartQA Dataset/translate_train/tables"
    # test_image_dir="/root/ChartQA/ChartQA Dataset/translate_test/png"
    # test_csv_dir="/root/ChartQA/ChartQA Dataset/translate_test/tables"
    # train_dataset = ImageCaptioningDataset(train_image_dir, train_csv_dir, processor)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)
    # test_dataset = ImageCaptioningDataset(test_image_dir, test_csv_dir, processor)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collator, num_workers=4)
    # epochs = 1000
    # train(model, epochs, train_dataloader)
    # test(model, test_dataloader)
    #============================#

    # for ChartQA & PlotQA & ChartToTable
    #============================#
    # chartqa_train_image_dir="/root/ChartQA/ChartQA Dataset/train/png"
    # chartqa_train_csv_dir="/root/ChartQA/ChartQA Dataset/train/tables"
    # chartqa_test_image_dir="/root/ChartQA/ChartQA Dataset/test/png"
    # chartqa_test_csv_dir="/root/ChartQA/ChartQA Dataset/test/tables"
    # plotqa_train_image_dir="/root/PlotQA/data/translated_train/png"
    # plotqa_train_csv_dir="/root/PlotQA/data/translated_train/csv"
    # plotqa_test_image_dir="/root/PlotQA/data/translated_test/png"
    # plotqa_test_csv_dir="/root/PlotQA/data/translated_test/csv"
    # charttotable_train_image_dir="/root/chart-to-table/data/train/png"
    # charttotable_train_txt_dir="/root/chart-to-table/data/train/txt"
    # charttotable_mix_train_image_dir="/root/chart-to-table-mix/data/train/png"
    # charttotable_mix_train_txt_dir="/root/chart-to-table-mix/data/train/txt"
    # charttotable_mix_test_image_dir="/root/chart-to-table-mix/data/test/png"
    # charttotable_mix_test_txt_dir="/root/chart-to-table-mix/data/test/txt"
    # chartqa_train_dataset = ImageCaptioningDataset(chartqa_train_image_dir, chartqa_train_csv_dir, processor)
    # plotqa_train_dataset = ImageCaptioningDataset(plotqa_train_image_dir, plotqa_train_csv_dir, processor)
    # charttotable_train_dataset = MixedChartToTableDataset(charttotable_train_image_dir, charttotable_train_txt_dir, processor)
    # charttotable_mix_train_dataset = MixedChartToTableDataset(charttotable_train_image_dir, charttotable_train_txt_dir, processor)
    # train_dataset = CombinedDataset(chartqa_train_dataset, plotqa_train_dataset, charttotable_train_dataset, charttotable_mix_train_dataset)
    # train_dataset = CombinedDataset(chartqa_train_dataset, plotqa_train_dataset)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)
    # train_dataloader = DataLoader(plotqa_train_dataset, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)
    # chartqa_test_dataset = ImageCaptioningDataset(chartqa_test_image_dir, chartqa_test_csv_dir, processor)
    # plotqa_test_dataset = ImageCaptioningDataset(plotqa_test_image_dir, plotqa_test_csv_dir, processor)
    # charttotable_mix_test_dataset = MixedChartToTableDataset(charttotable_mix_test_image_dir, charttotable_mix_test_txt_dir, processor)
    # test_dataset = CombinedDataset(chartqa_test_dataset, plotqa_test_dataset, charttotable_mix_test_dataset)
    # test_dataset = CombinedDataset(chartqa_test_dataset, plotqa_test_dataset)
    # chartqa_test_dataloader = DataLoader(chartqa_test_dataset, shuffle=False, batch_size=1, collate_fn=collator, num_workers=4)
    # plotqa_test_dataloader = DataLoader(plotqa_test_dataset, shuffle=False, batch_size=1, collate_fn=collator, num_workers=4)
    # epochs = 3
    # train(model, epochs, train_dataloader)
    # test(model, plotqa_test_dataloader)
    #============================#

    # for Public OCR
    #============================#
    # public_train_image_dir = "/root/공공행정문서 OCR/long_process_Validation/jpg"
    # public_train_json_dir = "/root/공공행정문서 OCR/long_process_Validation/json"
    # public_test_image_dir = "/root/공공행정문서 OCR/split_long_process_Validation/jpg"
    # public_test_json_dir = "/root/공공행정문서 OCR/split_long_process_Validation/json"
    # variable_train_image_dir = "/root/다양한 형태의 한글 문자 OCR/Chunk_Training/jpg"
    # variable_train_json_dir = "/root/다양한 형태의 한글 문자 OCR/Chunk_Training/json"
    # variable_test_image_dir = "/root/다양한 형태의 한글 문자 OCR/Chunk_Validation/jpg"
    # variable_test_json_dir = "/root/다양한 형태의 한글 문자 OCR/Chunk_Validation/json"
    # kostat_train_image_dir = "/root/inference/train_data/kostat/contents/train/image"
    # kostat_train_json_dir = "/root/inference/train_data/kostat/contents/train/json"
    # kostat_test_image_dir = "/root/inference/train_data/kostat/contents/test/image"
    # kostat_test_json_dir = "/root/inference/train_data/kostat/contents/test/json"
    # public_train_dataset = OCRDataset(public_train_image_dir, public_train_json_dir, processor)
    # variable_train_dataset = OCRDataset(variable_train_image_dir, variable_train_json_dir, processor)
    # kostat_train_dataset = OCRDataset(kostat_train_image_dir, kostat_train_json_dir, processor)
    # train_dataset = CombinedDataset(public_train_dataset, variable_train_dataset)
    # train_dataloader = DataLoader(kostat_train_dataset, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)
    # public_test_dataset = OCRDataset(public_test_image_dir, public_test_json_dir, processor)
    # variable_test_dataset = OCRDataset(variable_test_image_dir, variable_test_json_dir, processor)
    # kostat_test_dataset = OCRDataset(kostat_test_image_dir, kostat_test_json_dir, processor)
    # test_dataset = CombinedDataset(public_test_dataset, variable_test_dataset)
    # test_dataloader = DataLoader(kostat_test_dataset, shuffle=False, batch_size=1, collate_fn=collator, num_workers=4)
    # epochs = 5
    # train(model, epochs, train_dataloader)
    # test(model, test_dataloader)
    #============================#

    # for PlotQA & KostatOCR
    #============================#
    # plotqa_train_image_dir="/root/PlotQA/data/translated_train/png"
    # plotqa_train_csv_dir="/root/PlotQA/data/translated_train/csv"
    # plotqa_test_image_dir="/root/PlotQA/data/translated_test/png"
    # plotqa_test_csv_dir="/root/PlotQA/data/translated_test/csv"
    # kostat_train_image_dir = "/root/inference/train_data/kostat/contents/train/image"
    # kostat_train_json_dir = "/root/inference/train_data/kostat/contents/train/json"
    # kostat_test_image_dir = "/root/inference/train_data/kostat/contents/test/image"
    # kostat_test_json_dir = "/root/inference/train_data/kostat/contents/test/json"
    # plotqa_train_dataset = ImageCaptioningDataset(plotqa_train_image_dir, plotqa_train_csv_dir, processor)
    # plotqa_test_dataset = ImageCaptioningDataset(plotqa_test_image_dir, plotqa_test_csv_dir, processor)
    # kostat_train_dataset = OCRDataset(kostat_train_image_dir, kostat_train_json_dir, processor)
    # kostat_test_dataset = OCRDataset(kostat_test_image_dir, kostat_test_json_dir, processor)
    # train_dataset = CombinedDataset(plotqa_train_dataset, kostat_train_dataset)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)
    # test_dataset = CombinedDataset(plotqa_test_dataset, kostat_test_dataset)
    # test_dataloader = DataLoader(plotqa_test_dataset, shuffle=False, batch_size=1, collate_fn=collator, num_workers=4)
    # epochs = 2
    # train(model, epochs, train_dataloader)
    # test(model, test_dataloader)

    # for Partition
    #============================#
    # kostat_train_image_dir = "/root/part/part_image_cropped"
    # kostat_train_bbox_dir = "/root/part/part_txt"
    # kostat_train_part_dir = "/root/part/part_label"
    # kostat_test_image_dir = "/root/part/test_part_image_cropped"
    # kostat_test_bbox_dir = "/root/part/test_part_txt"
    # kostat_test_part_dir = "/root/part/test_part_label"

    # kostat_train_dataset = PartitionDataset(kostat_train_image_dir, kostat_train_bbox_dir, kostat_train_part_dir, processor)
    # kostat_test_dataset = PartitionDataset(kostat_test_image_dir, kostat_test_bbox_dir, kostat_test_part_dir, processor)
    # train_dataloader = DataLoader(kostat_train_dataset, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)
    # test_dataloader = DataLoader(kostat_test_dataset, shuffle=False, batch_size=1, collate_fn=collator, num_workers=4)
    # epochs = 50
    # train(model, epochs, train_dataloader)
    # test(model, test_dataloader)
    #============================#
    
    # train_image_dir="/root/deplot/only_graph"
    # train_csv_dir="/root/deplot/only_graph_label"
    # test_image_dir="/root/deplot/only_graph"
    # test_csv_dir="/root/deplot/only_graph_label"
    # train_dataset = ImageCaptioningDataset(train_image_dir, train_csv_dir, processor)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)
    # test_dataset = ImageCaptioningDataset(test_image_dir, test_csv_dir, processor)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collator, num_workers=4)
    # epochs = 50
    # train(model, epochs, train_dataloader)
    # test(model, test_dataloader)