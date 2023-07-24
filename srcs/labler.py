import re
from collections import Counter
from profanity import profanity
import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd

class SpamDetector:
    def __init__(self, profanity_threshold=2, length_threshold=2, fnwc_path=None, similarity_threshold=0.9):
        self.patterns = {
            "few_words": re.compile(r"^\s*\b\w+\b(?: \b\w+\b)?[.!?]?\s*$"),  
            "long_spaces": re.compile(r"\s{4,}"),
            "non_alpha_chars": re.compile(r"^[^A-Za-z\s]+$"),
            "no_variation": re.compile(r"^(\b\w+\b)( \1)*$")
        }
        self.spam_counts = Counter()
        self.max_consecutive_letters = 5
        self.profanity_threshold = profanity_threshold
        self.length_threshold = length_threshold
        self.similarity_threshold = similarity_threshold
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        # Load FnWCn dataset
        if fnwc_path is not None:## ADDED
            with open(fnwc_path, 'r') as f:
                self.fnwc_data = set(line.strip() for line in f)
        else:
            self.fnwc_data = set()

    def cosine_similarity(self, vec1, vec2):## ADDED
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return np.dot(vec1, vec2)

    def document_embedding(self, text):## ADDED
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Mean pooling - Take attention mask into account for correct averaging
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def check_profanity(self, text, thread_id):
        profane_words = text.split('*') # split the text by '*'
        profane_words = [word for word in profane_words if word != ''] 
        if len(profane_words) > self.profanity_threshold:
            return f"Profane threshold reached - original text stored by {thread_id}"
        else:
            return text

    def has_extra_letters(self, text):
        return re.search(r"([a-zA-Z])\1{%d,}" % self.max_consecutive_letters, text)

    def is_short_text(self, text):
        return len(text) <= self.length_threshold

    def Is_FnWC(self, text):
        return text in self.fnwc_data

    def is_similar_to_top(self, text, avg_relevant_vector):
        text_vector = self.document_embedding(text).numpy()
        similarity = self.cosine_similarity(text_vector, avg_relevant_vector)
        return similarity >= self.similarity_threshold

    def label_spam(self, text, thread_id, avg_relevant_vector):

        text = text.lower()
        text = self.check_profanity(text, thread_id)
        if text.startswith("Profane threshold reached"):
            return text
        if any(pattern.search(text) for pattern in self.patterns.values()):
            self.spam_counts["regex_patterns"] += 1
            return "SPAM"
        if self.has_extra_letters(text):
            self.spam_counts["extra_letters"] += 1
            return "SPAM"  
        if self.Is_FnWC(text):
            self.spam_counts["FnWC"] += 1
            return "SPAM"
        if self.is_short_text(text):
            self.spam_counts["short_text"] += 1
            return "SPAM"
            ## ADDED
        if not self.is_similar_to_top(text, avg_relevant_vector): 
            self.spam_counts["not_similar_to_top"] += 1
            return "SPAM"
        return "NOT_SPAM"

def apply_detector(data,sample_size, similarity_threshold):
    def apply_spam_detector(row, detector, avg_vector):
        try:
            text = row['text_clean']
            thread_id = row['thread_id']
            result = detector.label_spam(text, thread_id, avg_vector)
            if result.startswith("Profane threshold reached"):
                row['text_clean'] = result
                row['spam_label'] = "NOT_SPAM"
            else:
                row['spam_label'] = result
        except Exception as e:
            row['spam_label'] = 'ERROR'
            print(f'Error processing row {row.name}: {e}')
        return row

    def label_spam(data, detector, avg_vector):
        return data.apply(apply_spam_detector, detector=detector, avg_vector=avg_vector, axis=1)
    
    def get_spam_ratio(data, label_column):
        return data[label_column].value_counts(normalize=True)
    file_path = './.samples/'
    text_column = 'text_clean'
    top_text_column = '/data_drive/processed/freq_logging/top/'
    spam_detector = SpamDetector(profanity_threshold=3, length_threshold=2, 
                                 fnwc_path='/data_drive/processed/freq_logging/top/text_clean.txt', 
                                 similarity_threshold=similarity_threshold)

    # Read the 'final_top.txt' file
    final_top_df = pd.read_csv(os.path.join(top_text_column, 'final_top.txt'))
    # Filter relevant texts
    relevant_texts = final_top_df['text_clean']

    # Compute document embeddings for the texts
    relevant_vectors = relevant_texts.apply(lambda x: spam_detector.document_embedding(x))

    # Compute the average vector
    avg_relevant_vector = np.mean(np.vstack(relevant_vectors.values), axis=0)

    # Label the data as SPAM or NOT_SPAM
    labeled_data = label_spam(data.sample(sample_size), spam_detector, avg_relevant_vector)
    # Calculate the ratio of SPAM to total
    spam_ratio = get_spam_ratio(labeled_data, 'spam_label')
    
    return spam_ratio, labeled_data, spam_detector
