import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Sentence-BERT model
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def determine_optimal_k(embeddings, max_k=10):
    """Use the Elbow Method to determine the optimal number of clusters"""
    inertia_values = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(embeddings)
        inertia_values.append(kmeans.inertia_)
    
    # Plot the Elbow Method graph
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, inertia_values, 'bo-', markersize=8)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    # Determine the elbow point (you can do this manually or automate it)
    # Here, you can select k by visual inspection or use more sophisticated methods to find the "elbow"
    optimal_k = int(input("Enter the optimal number of clusters (k) based on the Elbow Method plot: "))
    return optimal_k

def process_scenario(scenario, total_human_responses=118):
    # Read and load data
    human_responses_all = json.load(open('translated_responses.json', 'r'))
    human_need_detection = human_responses_all[f'need {scenario}']
    human_solution = human_responses_all[f'solution {scenario}']
    robot_response = json.load(open(f'scenario{scenario}_need_solution.json', 'r'))
    needs = robot_response['Environment']['human']['needs']
    robot_need_detection = [value['description'] for key, value in needs.items()]
    robot_solution = [value['suggested robot solution'] for key, value in needs.items()]

    # Process needs
    process_and_visualize(scenario, 'Need', human_need_detection, robot_need_detection, total_human_responses)
    
    # Process solutions
    process_and_visualize(scenario, 'Solution', human_solution, robot_solution, total_human_responses)

def process_and_visualize(scenario, response_type, human_responses, robot_responses, k_value, total_human_responses):
    # Compute embeddings
    human_embeddings = embedding_model.encode(human_responses, convert_to_tensor=True)
    robot_embeddings = embedding_model.encode(robot_responses, convert_to_tensor=True)

    # Move embeddings from GPU/MPS to CPU
    human_embeddings = human_embeddings.cpu().numpy() if torch.is_tensor(human_embeddings) else human_embeddings
    robot_embeddings = robot_embeddings.cpu().numpy() if torch.is_tensor(robot_embeddings) else robot_embeddings

    # Determine the optimal number of clusters using the Elbow Method
    k_value = determine_optimal_k(human_embeddings)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k_value, random_state=0)
    clusters = kmeans.fit_predict(human_embeddings)

    # Reduce embeddings to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(human_embeddings)
    robot_embedding_reduced = pca.transform(robot_embeddings)

    # Predict the clusters for robot-generated responses using the trained KMeans model
    robot_labels = kmeans.predict(robot_embeddings)

    # Extract top keywords for each cluster
    clustered_texts = {}
    for cluster_id, text in zip(clusters, human_responses):
        if cluster_id not in clustered_texts:
            clustered_texts[cluster_id] = []
        clustered_texts[cluster_id].append(text)

    def extract_top_keywords(texts, top_n=3):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
        sorted_indices = np.argsort(avg_tfidf)[::-1]
        top_keywords = [feature_names[index] for index in sorted_indices[:top_n]]
        return top_keywords

    cluster_keywords = {}
    for cluster_id, texts in clustered_texts.items():
        cluster_keywords[cluster_id] = extract_top_keywords(texts)

    # Calculate cosine similarity and proportion rate
    average_cosine_similarities = []
    proportion_rates = []
    for i, robot_embedding in enumerate(robot_embeddings):
        cluster_id = robot_labels[i]
        cluster_embeddings = human_embeddings[clusters == cluster_id]
        similarities = util.cos_sim(robot_embedding, cluster_embeddings)
        avg_similarity = similarities.mean().item()
        average_cosine_similarities.append(avg_similarity)
        
        cluster_size = np.sum(clusters == cluster_id)
        proportion_rate = cluster_size / total_human_responses
        proportion_rates.append(proportion_rate)
        

    # Visualize clustering results
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    for cluster_id, label in cluster_keywords.items():
        cluster_points = reduced_embeddings[np.array(clusters) == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        plt.text(centroid[0], centroid[1], label, fontsize=10, color='black', fontweight='bold')

    # Plot robor=t-generated embeddings and display similarity and proportion rate
    for i, (x, y) in enumerate(robot_embedding_reduced):
        plt.scatter(x, y, c='red', marker='x', s=100)
        plt.text(x, y, f'robot {i+1}\nSim: {average_cosine_similarities[i]:.2f}\nProp: {proportion_rates[i]:.2f}', 
                 fontsize=10, color='blue', fontweight='bold')

    plt.legend(loc="upper right")
    plt.title(f'Clustering of {response_type} Responses (Scenario {scenario})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(f'scenario_{scenario}_{response_type.lower()}_clustering.png')
    plt.close()

# Batch process all scenarios
for scenario in range(1, 17):
    process_scenario(scenario, k_value=4, total_human_responses=118)
