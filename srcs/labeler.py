import re
from collections import Counter
from collections.abc import Iterable
from profanity import profanity
import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd

class DialogDetector:
    # DialogParser
    ## Initialization: Chanscope is initialized with predefined patterns and thresholds.
    def __init__(self, profanity_threshold=3,length_threshold=2, fnwc_path=None, similarity_threshold=0.5, 
                 disable_profanity_filter=True, disable_storing_above_threshold=False):
        self.patterns = {
            "few_words": re.compile(r"^\s*\b\w+\b(?: \b\w+\b)?[.!?]?\s*$"),  
            "long_spaces": re.compile(r"\s{4,}"),
            "non_alpha_chars": re.compile(r"^[^A-Za-z\s]+$"),
            "no_variation": re.compile(r"^(\b\w+\b)( \1)*$")
        }
        self.spam_counts = Counter()
        self.max_consecutive_letters = 4
        self.profanity_threshold = profanity_threshold
        self.length_threshold = length_threshold
        self.similarity_threshold = similarity_threshold
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        self.disable_profanity_filter = disable_profanity_filter
        self.disable_storing_above_threshold = disable_storing_above_threshold

        if fnwc_path is not None:
            ## Load FnWCn dataset
            with open(fnwc_path, 'r') as f:
                self.fnwc_data = set(line.strip() for line in f)
        else:
            self.fnwc_data = set()

    def is_dialog_comment(self, posted_comment, thread_id):
        references = re.findall(r'&gt;&gt;\d+', posted_comment)
        thread_id = int(thread_id)  # Ensure proper comparison
        for reference in references:
            ref_id = reference.replace('&gt;&gt;', '')
            if ref_id != thread_id:
                return True
        return False
    
    def cosine_similarity(self, vec1, vec2):
        ## Comparing Texts to a Reference Vector: The vectors for the texts are compared
        ## to an average reference vector. This comparison is done using cosine similarity.
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return np.dot(vec1, vec2)

    def document_embedding(self, text):
        ## Vectorizing the Texts: Each text is transformed into a fixed-size vector
        ## that represents the semantic content of the text. This is done using BERT.
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        ## Mean pooling - Take attention mask into account for correct averaging
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def check_profanity(self, text, thread_id):
        profane_words = text.split('*')  # Split the text by '*'
        profane_words = [word for word in profane_words if word.strip() != '']  # Remove empty strings
        if len(profane_words) - 1 > self.profanity_threshold:  # Subtract 1 to count the number of '*' characters
            return f"Profane threshold reached - original text stored by {thread_id}"
        else:
            return text

    def has_extra_letters(self, text):
        return re.search(r"([a-zA-Z])\1{%d,}" % self.max_consecutive_letters, text)
    def is_short_text(self, text):
        return len(text) <= self.length_threshold
    def Is_FnWC(self, text):
        return text in self.fnwc_data

    def is_similar(self, text, avg_spam_vector, avg_not_spam_vector):
        text_vector = self.document_embedding(text).numpy()
        similarity_with_spam = self.cosine_similarity(text_vector, avg_spam_vector)
        similarity_with_not_spam = self.cosine_similarity(text_vector, avg_not_spam_vector)

        if similarity_with_spam > self.similarity_threshold:
            return "SPAM"
        elif similarity_with_not_spam > self.similarity_threshold:
            return "NOT_SPAM"
        else:
            return ""

    def label_spam(self, text, thread_id, avg_spam_vector, avg_not_spam_vector, posted_comment, enable_dialog_comment_check=False):
        # Making a Decision Based on Similarity: If the similarity between a text's vector
        # and the reference vector is above a threshold, the text is considered relevant (not spam).
        # If the similarity is below the threshold, the text is considered spam.
        if enable_dialog_comment_check and self.is_dialog_comment(posted_comment, thread_id):
            return "NOT_SPAM"
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
        label_based_on_similarity = self.is_similar(text, avg_spam_vector, avg_not_spam_vector)
        if label_based_on_similarity == "SPAM":
            self.spam_counts["similar_to_spam"] += 1
        elif label_based_on_similarity == "NOT_SPAM":
            self.spam_counts["similar_to_not_spam"] += 1
        else:
            self.spam_counts["not_similar_to_either"] += 1
        return label_based_on_similarity
        
def apply_detector(data, similarity_threshold, sample_size = None,
                   disable_profanity_filter=True, disable_storing_above_threshold=False,
                   enable_dialog_comment_check=False): # Added new parameter

    def apply_spam_detector(row, detector, avg_spam_vector, avg_not_spam_vector):
        try:
            text = row['text_clean']
            posted_comment = row['posted_comment']
            thread_id = row['thread_id']
            result = detector.label_spam(text, thread_id, avg_spam_vector, avg_not_spam_vector, posted_comment, enable_dialog_comment_check)
            row['is_dialog'] = detector.is_dialog_comment(posted_comment, thread_id) 
            if result.startswith("Profane threshold reached"):
                row['text_clean'] = result
                row['spam_label'] = "NOT_SPAM"
            else:
                row['spam_label'] = result
        except Exception as e:
            row['spam_label'] = 'ERROR'
            print(f'Error processing row {row.name}: {e}')
        return row

    def label_spam(data, detector, avg_spam_vector, avg_not_spam_vector):
        return data.apply(apply_spam_detector, detector=detector, avg_spam_vector=avg_spam_vector, avg_not_spam_vector=avg_not_spam_vector, axis=1)

    def get_spam_ratio(data, label_column):
        return data[label_column].value_counts(normalize=True)

    # Supporting file paths
    file_path = './data/datasets/baselines/4chan/'
    text_column = 'text_clean'
    fnwc_path='/data_drive/processed/freq_logging/top/text_clean.txt' 
    top_text_column = '/data_drive/processed/freq_logging/top/'
    # Initialize the Spam Detector
    spam_detector = DialogDetector(profanity_threshold=3, length_threshold=2, 
                                fnwc_path=fnwc_path,
                                similarity_threshold=similarity_threshold,
                                disable_profanity_filter=disable_profanity_filter,
                                disable_storing_above_threshold=disable_storing_above_threshold)
    # Read the spam dataset
    spam_dataset_df = pd.read_parquet(os.path.join(file_path, 'spam_dataset.parquet')).dropna(subset=[text_column])
    final_top_df = pd.read_csv(os.path.join(top_text_column, 'final_top.txt'))

    # Separate the dataset into SPAM and NOT_SPAM based on spam_label
    spam_set = spam_dataset_df[spam_dataset_df['spam_label'] == 'SPAM']['text_clean']
    not_spam_set = spam_dataset_df[spam_dataset_df['spam_label'] == 'NOT_SPAM']['text_clean']

    top_freq = final_top_df['text_clean'].drop_duplicates()
    top_freq_df = pd.DataFrame({
        'text_clean': final_top_df['text_clean'].drop_duplicates(),
        'label': 'SPAM'
    })

    # Compute document embeddings for SPAM texts and NOT_SPAM texts
    spam_vectors = spam_set.apply(lambda x: spam_detector.document_embedding(x))        
    not_spam_vectors = not_spam_set.apply(lambda x: spam_detector.document_embedding(x))

    # Compute the average vector for SPAM and NOT_SPAM, serving as a reference or baseline for similarity checks
    avg_spam_vector = np.mean(np.vstack(spam_vectors.values), axis=0)
    avg_not_spam_vector = np.mean(np.vstack(not_spam_vectors.values), axis=0)

    # Label the remaining data as SPAM or NOT_SPAM using the computed average vectors
    labeled_data = label_spam(data.sample(sample_size), spam_detector, avg_spam_vector, avg_not_spam_vector)

    # Calculate the ratio of SPAM to total in the labeled data
    spam_ratio = get_spam_ratio(labeled_data, 'spam_label')
    labeled_data['spam_label'] = labeled_data['spam_label'].apply(lambda x: 1 if x == 'SPAM' else 0)
    similarity_training_data = labeled_data.copy()
    similarity_training_data = similarity_training_data[similarity_training_data['spam_label'] != '']
    return spam_ratio, labeled_data, similarity_training_data, spam_detector
