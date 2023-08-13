import re
import configparser
import warnings
import string
from urllib.parse import urlparse
from bs4 import BeautifulSoup,MarkupResemblesLocatorWarning
from unicodedata import normalize
import pandas as pd
import numpy as np
from profanity import profanity
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

contraction_mapping = pd.read_json('utils/contraction_mapping.json', typ='series').to_dict()

## Function to pad punctuation in a given string
def pad_punctuation(s):
    s = remove_urls(s)
    s = remove_whitespace(s)
    s = re.sub(f"([{string.punctuation}])", r"\1", s)
    s = re.sub(" +", " ", s)
    return s.strip()

# Function to remove URLs from a text string
def remove_urls(text):
    if isinstance(text, str):
        urls = re.findall(r'http\S+|www.\S+', text)
        for url in urls:
            base = urlparse(url).netloc
            base = re.sub(r'^www\.', '', base)
            text = text.replace(url, base)
        text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        soup = BeautifulSoup(text, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        text = re.sub(r'>>\d+', ' ', text)
        text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
        text = re.sub(r'[^a-zA-Z0-9.,!?\' ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return text

## Function to remove excessive white spaces
def remove_whitespace(text):
    return " ".join(text.split())

## Function to remove profanity from a given text
def remove_profanity(text):
  words = text.split()
  cleaned_words = [("*" * len(word)) if profanity.contains_profanity(word) else word for word in words]
  return " ".join(cleaned_words)

## Combined text cleaning function
def clean_text(text):
  text = remove_urls(text)
  text = remove_profanity(text)
  text = remove_whitespace(text)
  return text

## Function to change the spam label of certain thread IDs in a DataFrame
def flip_spam_label(df, thread_ids):
    if not isinstance(thread_ids, list):
        thread_ids = [thread_ids]
    for thread_id in thread_ids:
        idx = df[df['thread_id'] == thread_id].index
        current_label = df.loc[idx, 'spam_label'].values[0]
        new_label = 'SPAM' if current_label == 'NOT_SPAM' else 'NOT_SPAM'
        df.loc[idx, 'spam_label'] = new_label