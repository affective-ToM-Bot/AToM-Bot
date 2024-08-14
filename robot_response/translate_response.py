import pandas as pd
import re
import json
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import os

# Load the CSV file into a DataFrame
response = pd.read_csv('open_vocabulary_response.csv')

# Set source and target languages
src_lang = 'zh'
tgt_lang = 'en'
model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

# Load the translation model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def custom_segment(text):
    """Segment text using Chinese commas, spaces, and commas"""
    segments = re.split(r'[、，,;； ]+', text)
    return [seg for seg in segments if seg]  # Remove empty strings

def translate(texts):
    """Translate a list of texts"""
    translated = model.generate(**tokenizer(texts, return_tensors="pt", padding=True, truncation=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

def process_column(column):
    """Process a column by segmenting and translating its content"""
    human_responses = [str(item) for item in column]
    
    all_segments = []
    for response in human_responses:
        all_segments.extend(custom_segment(response))
    
    translated_human_answers = translate(all_segments)
    return translated_human_answers

# Initialize a dictionary to store processed columns
translated_data = {}

# Check if a partial output file exists
output_file = 'translated_responses.json'
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        translated_data = json.load(f)

# Process each column and save immediately
for col in tqdm(response.columns, desc="Processing columns"):
    if col not in translated_data:
        translated_data[col] = process_column(response[col])
        # Save the progress after each column is processed
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=4)

print(f"Translation and processing completed, saved to '{output_file}'.")