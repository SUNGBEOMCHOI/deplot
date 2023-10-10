from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

processor = Pix2StructProcessor.from_pretrained('./final_model')
model = Pix2StructForConditionalGeneration.from_pretrained('./chartqa+plotqa_1').to('cuda')

# url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("./스크린샷 2023-10-11 022106.png")

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt").to('cuda')
predictions = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(predictions[0], skip_special_tokens=True))
