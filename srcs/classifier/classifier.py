import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer
import torch
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np

model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_name)

def flatten(lis):
    for item in lis:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

# Document embedding function
def document_embedding_Five_Number_Summary(text, max_length=512):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    attentions = model_output['attentions']
    last_layer_attentions = attentions[-1]
    last_layer_attentions_flat = last_layer_attentions.squeeze(0).reshape(last_layer_attentions.size(1), -1)
    result = []
    for head in range(last_layer_attentions_flat.shape[0]):
        # Sort the attention values to facilitate the extraction of the five-number summary
        sorted_attention_values = torch.sort(last_layer_attentions_flat[head])[0]
        min_val = sorted_attention_values[0].item()
        q1 = torch.quantile(sorted_attention_values, 0.25).item()
        median = torch.median(sorted_attention_values).item()
        q3 = torch.quantile(sorted_attention_values, 0.75).item()
        max_val = sorted_attention_values[-1].item()
        # Add the five-number summary to the result
        result.extend([min_val, q1, median, q3, max_val])
    # Pad the result with zeros to match max_length (ensures consistent length across documents)
    padded_result = result + [0] * (max_length - len(result))
    return padded_result

def load_and_process_data(split_data, split_list=['train', 'val', 'test']):
    data_dict = {}
    max_length = 512
    # Check if split_data is a string (potential file path) or a DataFrame
    if isinstance(split_data, str):
        # Load data from file path
        for split in split_list:
            data = pd.read_parquet(f'{split_data}/{split}.parquet')
            data.fillna('', inplace=True)
            data['vector'] = data['text_clean'].apply(lambda x: document_embedding_Five_Number_Summary(x, max_length))
            max_length = max(max_length, max(data['vector'].apply(len)))
            data['vector'] = data['vector'].apply(lambda x: list(flatten(x)))
            data_dict[split] = data
    elif isinstance(split_data, pd.DataFrame):
        # Process data from DataFrame
        data = split_data.copy()
        data.fillna('', inplace=True)
        data['vector'] = data['text_clean'].apply(lambda x: document_embedding_Five_Number_Summary(x, max_length))
        max_length = max(max_length, max(data['vector'].apply(len)))
        data['vector'] = data['vector'].apply(lambda x: list(flatten(x)))
        data_dict = {'df': data}
    else:
        raise TypeError("split_data must be a file path (string) or a DataFrame.")
    return data_dict, max_length

def create_model(input_shape, layers, dropout, batch_normalization, l1_reg, l2_reg):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layers[0], activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
    if batch_normalization:
        model.add(tf.keras.layers.BatchNormalization())
    for l in layers[1:]:
        model.add(tf.keras.layers.Dense(l, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
        if batch_normalization:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def train_model(train_data, train_labels, val_data, val_labels, params):
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Define the model save path with .keras extension
    
    model_save_path = os.path.join(save_dir, 'best_model.tf')
    # Train the model
    model = create_model((train_data.shape[1],), params['layers'], params['dropout'], params['batch_normalization'],
                         params['l1_reg'], params['l2_reg'])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params['initial_learning_rate'],
        decay_steps=params['decay_steps'],
        decay_rate=params['decay_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(), 
                  metrics=['accuracy'])
    # Set callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.01, patience=10, verbose=0, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')
    callbacks = [reduce_lr, early_stopping, model_checkpoint]
    # Adjust the number of epochs
    history = model.fit(train_data, train_labels, epochs=params['epochs'],
                        validation_data=(val_data, val_labels), callbacks=callbacks)
    return history, model

