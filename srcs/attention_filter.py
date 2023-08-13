from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable

# Recursive function to flatten nested lists
def flatten(lis):
    # Iterate through the input list
    for item in lis:
        # If the item is an Iterable (such as a list or tuple) and not a string
        if isinstance(item, Iterable) and not isinstance(item, str):
            # Recursively call the flatten function for the sub-item
            for x in flatten(item):
                yield x
        else:
            # If the item is not an Iterable or is a string, yield the item as-is
            yield item

def find_elbow(data, max_k):
    iters = range(2, max_k+1, 2)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(data)
        sse.append(kmeans.inertia_)
    slopes = [sse[i] - sse[i - 1] for i in range(1, len(sse))]
    elbow = slopes.index(max(slopes))
    return iters[elbow]

def document_embedding_Five_Number_Summary(text, max_length):
    model_name = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Tokenize the text, pad it to a consistent length, and truncate it if necessary
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        # Pass the encoded input through the model to obtain the embeddings
        model_output = model(**encoded_input)

    # Get the attention values from the output
    attentions = model_output['attentions']
    # Extract the attention values from the last layer
    last_layer_attentions = attentions[-1]
    # Flatten the last dimension to treat it as a one-dimensional distribution for each head
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

def main_plotting(data, max_length=512, show_scatter=False):
    data['vector'] = data['text_clean'].apply(document_embedding_Five_Number_Summary, max_length=max_length)
    data['vector'] = data['vector'].apply(lambda x: list(flatten(x)))
    max_length = max(data['vector'].apply(len))
    data['vector'] = data['vector'].apply(lambda x: np.pad(x, (0, max_length - len(x))))
    X = np.vstack(data['vector'].values)
    opt = find_elbow(X, 20)
    kmeans = KMeans(n_clusters=opt, random_state=42, n_init=10).fit(X)
    data['cluster'] = kmeans.labels_
    cluster_junk_ratio = data.groupby('cluster')['spam_label'].mean()
    
    if show_scatter == False:
        return data, cluster_junk_ratio

    print(cluster_junk_ratio)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    data['pca-one'] = pca_result[:, 0]
    data['pca-two'] = pca_result[:, 1]
    centroids_pca = pca.transform(kmeans.cluster_centers_)  # Apply PCA to the cluster centroids
    
    fig, ax = plt.subplots(figsize=(16, 10))
    sorted_clusters = sorted(data['cluster'].unique())
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(sorted_clusters))]
    
    for idx, cluster in enumerate(sorted_clusters):
        cluster_data = data[data['cluster'] == cluster]
        for is_spam in cluster_data['spam_label'].unique():
            subset = cluster_data[cluster_data['spam_label'] == is_spam]
            label = f"Cluster {cluster} - {'Spam' if is_spam else 'Not Spam'}"
            ax.scatter(subset['pca-one'], subset['pca-two'], color=colors[idx], label=label, alpha=0.7 if is_spam else 1)
        
        # Plot cluster centroids
        ax.scatter(centroids_pca[idx, 0], centroids_pca[idx, 1], marker='x', color='red')
        
        # Annotate with junk ratio
        ax.annotate(f"Junk Ratio: {cluster_junk_ratio.loc[cluster]:.2f}",
                    (centroids_pca[idx, 0], centroids_pca[idx, 1]), fontsize=5, color='black')

    ax.set_title('Clusters after PCA transformation') # Use ax for setting title
    handles, labels = ax.get_legend_handles_labels() # Get legend handles and labels from ax
    sorted_legend = sorted(zip(labels, handles), key=lambda x: int(x[0].split(" ")[1]))
    ax.legend([handle for label, handle in sorted_legend],
            [label for label, handle in sorted_legend]) # Use ax for legend
    return fig, cluster_junk_ratio, data
