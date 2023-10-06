import os
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
        image = Image.open(image_path).convert("RGB")

        # Load csv content and replace commas with pipes
        csv_path = os.path.join(self.csv_dir, self.csv_filenames[idx])
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # Don't replace commas in the header
                line_parts = line.split(',')
                line = ' | '.join(line_parts)
                line = line.replace('\n', ' <0x0A> ')
                lines[i] = line

            text = ''.join(lines)

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
        assert len(self.image_filenames) == len(self.json_filenames), "Number of images and CSV files do not match!"

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

def collator(batch):
    new_batch = {"flattened_patches":[], "attention_mask":[]}
    texts = [item["text"] for item in batch]

    text_inputs = processor(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=200, truncation=True)

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
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_dataset)*epochs*0.15), num_training_steps=len(train_dataset)*epochs)
    # scheduler = AdafactorSchedule(optimizer)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    model.train()

    for epoch in range(epochs):
        print("Epoch:", epoch)
        total_loss = 0
        for idx, batch in enumerate(tqdm.tqdm(train_dataloader)):
            with accelerator.accumulate(model):
                # labels = batch.pop("labels").to(device)
                # flattened_patches = batch.pop("flattened_patches").to(device)
                # attention_mask = batch.pop("attention_mask").to(device)
                labels = batch.pop("labels")
                flattened_patches = batch.pop("flattened_patches")
                attention_mask = batch.pop("attention_mask")
                
                outputs = model(flattened_patches=flattened_patches,
                                attention_mask=attention_mask,
                                labels=labels)

                loss = outputs.loss
                # loss = loss / gradient_accumulation_steps # +
                # print("Loss:", loss.item())
                total_loss += loss.item()


                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
                accelerator.backward(loss)
                # if (idx+1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                

                # scheduler.step()
                if (epoch + 1) % 2 == 0:
                    model.eval()

                    predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)
                    print("---------Predictions---------\n", processor.batch_decode(predictions, skip_special_tokens=True))
                    print()
                    print("---------labels---------\n", batch.pop("raw_labels")[0])

                    model.train()
        if (epoch + 1) % 4 == 0:
            model.save_pretrained(f'{epoch+1}_model')
        with open('loss.txt', 'a') as f:
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

        predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens=512)
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
    model = Pix2StructForConditionalGeneration.from_pretrained('./final_model').to(device)
    processor.image_processor.is_vqa = False

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

    # for Public OCR
    #============================#
    train_image_dir="/root/공공행정문서 OCR/process_Validation/jpg"
    train_json_dir="/root/공공행정문서 OCR/process_Validation/json"
    test_image_dir="/root/공공행정문서 OCR/split_process_Validation/jpg"
    test_json_dir="/root/공공행정문서 OCR/split_process_Validation/json"
    train_dataset = OCRDataset(train_image_dir, train_json_dir, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator, num_workers=4)
    test_dataset = OCRDataset(test_image_dir, test_json_dir, processor)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collator, num_workers=4)
    epochs = 20
    train(model, epochs, train_dataloader)
    test(model, test_dataloader)
    #============================#
    