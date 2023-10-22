from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

processor = Pix2StructProcessor.from_pretrained('./final_model')
# model = Pix2StructForConditionalGeneration.from_pretrained('./final_model').to('cuda')
# model = Pix2StructForConditionalGeneration.from_pretrained('./kostat_ocr_model').to('cuda')
# model = Pix2StructForConditionalGeneration.from_pretrained('./chartqa_plotqa_5').to('cuda')
model = Pix2StructForConditionalGeneration.from_pretrained('./2_model').to('cuda')

# url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("/root/inference/test_data/image/2023년+상반기+지역별고용조사+취업자의+산업+및+직업별+특성 (1).pdf/1_1_contents.png")

# inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt").to('cuda')
inputs = processor(images=image, text="Optical character recognition below:", return_tensors="pt").to('cuda')
# inputs = processor(images=image, text="", return_tensors="pt").to('cuda')
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
