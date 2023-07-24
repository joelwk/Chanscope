import re
import configparser
import warnings
from urllib.parse import urlparse
from bs4 import BeautifulSoup,MarkupResemblesLocatorWarning
from unicodedata import normalize
import numpy as np
from profanity import profanity
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
config_transformer = configparser.ConfigParser()

config_transformer.read('./.config.ini')
board_info = config_transformer["board_info"]
source_data_s3 = config_transformer["source_data_s3"]
s3_bucket_name = source_data_s3['s3_bucket']
s3_bucket_data = source_data_s3['s3_bucket_data']
s3_bucket_processed = source_data_s3['s3_bucket_processed']
processed_data_gz = source_data_s3['processed_data_gz']
s3_bucket_batchprocessed = source_data_s3['s3_bucket_batchprocessed']

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                               "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                               "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                               "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                               "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                               "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                               "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                               "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                               "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                               "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                               "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                               "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                               "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                               "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                               "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                               "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                               "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                               "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                               "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                               "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                               "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                               "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                               "you're": "you are", "you've": "you have"}  

def remove_urls(text):
    if isinstance(text, str):
       # text = text.lower()
        urls = re.findall(r'http\S+|www.\S+', text)
        for url in urls:
            base = urlparse(url).netloc
            base = re.sub(r'^www\.', '', base)
            base = re.sub(r'^http://', '', base)
            base = re.sub(r'^https://', '', base)
            text = text.replace(url, f'{base}') 

        text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        soup = BeautifulSoup(text, 'html.parser')
        text = ' '.join([t for t in soup.stripped_strings])
        text = re.sub(r'>>\d+', ' ', text)
        text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
        text = re.sub(r'[^a-zA-Z0-9.,!?\' ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return text

def remove_whitespace(text):
    return " ".join(text.split())

def remove_profanity(text):
  words = text.split()
  cleaned_words = [("*" * len(word)) if profanity.contains_profanity(word) else word for word in words]
  return " ".join(cleaned_words)

def clean_text(text):
  text = remove_urls(text)
  text = remove_whitespace(text)
  text = remove_profanity(text)
  return text

def remove_empty_space(data, columns):
    for column in columns:
        data[column] = data[column].replace(['', ' '], np.nan)
        data = data.dropna(subset=[column])
    return data
    
def flip_spam_label(df, thread_ids):
    if not isinstance(thread_ids, list):
        thread_ids = [thread_ids]
    for thread_id in thread_ids:
        idx = df[df['thread_id'] == thread_id].index
        current_label = df.loc[idx, 'spam_label'].values[0]
        new_label = 'SPAM' if current_label == 'NOT_SPAM' else 'NOT_SPAM'
        df.loc[idx, 'spam_label'] = new_label