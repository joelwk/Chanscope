











import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def count_matches(input_str, column_str):
    if pd.notna(column_str):
        count = 0
        input_words = set(input_str.lower().split())
        column_words = set(column_str.lower().split())
        matching_words = input_words.intersection(column_words)
        count = len(matching_words)
        return count
    return 0

def calculate_similarity(input_str, column_str):
    if pd.notna(input_str) and pd.notna(column_str) and input_str.strip() and column_str.strip():
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([input_str, column_str])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]
    return 0

best_weights = {
    'summary_count': 1,
    'related_count': 1,
    'similarity_score': 0.051
}

def weighted_harmonic_mean_scoring_function(summary_count, related_count, similarity_score, weights):
    weighted_values = [
        (weights['summary_count'], summary_count),
        (weights['related_count'], related_count),
        (weights['similarity_score'], similarity_score)
    ]

    numerator = sum(w * v for w, v in weighted_values)
    denominator = sum(w / (v + 1e-8) for w, v in weighted_values)

    return numerator / denominator if denominator != 0 else 0

# The rest of the code remains the same

def check_similarity_threshold(current_similarity, original_similarity, threshold_percentage):
    threshold = threshold_percentage * original_similarity
    return abs(current_similarity - original_similarity) <= threshold

results = []
for index, input_row in training_test.iterrows():
    input_string = input_row['text_clean']
    max_composite_score = -1
    best_topic = None
    best_summary_count = 0
    best_related_count = 0
    best_similarity_score = 0

    for _, train_row in train_trends.iterrows():
        topic = train_row['Topic']
        summary_count = count_matches(input_string, train_row['Summary'])
        related_count = count_matches(input_string, train_row['Related News'])
        similarity_score = calculate_similarity(input_string, train_row['Summary'])

        original_similarity_score = calculate_similarity(input_string, train_trends.loc[train_trends['Topic'] == topic, 'Summary'].iloc[0])

        if check_similarity_threshold(similarity_score, original_similarity_score, 0.25):
            composite_score = summary_count + related_count + similarity_score

            if composite_score > max_composite_score:
                max_composite_score = composite_score
                best_topic = topic
                best_summary_count = summary_count
                best_related_count = related_count
                best_similarity_score = similarity_score

    if best_topic is not None:
        results.append([best_topic, input_string, best_summary_count, best_related_count, best_similarity_score])

result_headers = ['Best Topic', 'Input Text', 'Summaries', 'Related', 'Similarity']
trending_results_df = pd.DataFrame(results, columns=result_headers)
