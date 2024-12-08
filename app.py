from flask import Flask, render_template, request, send_from_directory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from PIL import Image
import os
from open_clip import create_model_and_transforms, tokenizer
import open_clip
import torch.nn.functional as F
import torch
import pickle
import pandas as pd

app = Flask(__name__)
file_prefix =  "/Users/ayeshanaeem/Downloads/coco_images_resized/"

@app.route("/images/<path:filename>")
def custom_image(filename):
    return send_from_directory(file_prefix, filename)
# Load images and embeddings
image_dir = "/Users/ayeshanaeem/Downloads/coco_images_resized"  # Update this to your folder path
image_embeddings = []  # List of embeddings for the images
image_names = []  # List of image filenames

# Load precomputed embeddings and filenames (Assuming stored as .npy)
df = pd.read_pickle("/Users/ayeshanaeem/Downloads/image_embeddings.pickle")

# Load CLIP model
clip_model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
clip_model.eval()

def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

def load_image(image_path, target_size=(224,224)):
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize(target_size)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    return img_array.flatten()

def pca_train_images(train_images, k = 50):
    # Apply PCA
    pca = PCA(n_components=k)
    pca.fit(train_images)  # Fit PCA on the training subset
    return pca

def pca_transform_images(transform_images, pca):
    reduced_embeddings = pca.transform(transform_images)
    return reduced_embeddings

def nearest_neighbors(query_embedding, embeddings, top_k=5):
    # query_embedding: The embedding of the query item (e.g., the query image) in the same dimensional space as the other embeddings.
    # embeddings: The dataset of embeddings that you want to search through for the nearest neighbors.
    # top_k: The number of most similar items (nearest neighbors) to return from the dataset.
    # Hint: flatten the "distances" array for convenience because its size would be (1,N)
    distances = euclidean_distances(np.array([query_embedding]), embeddings)
    distances = distances.flatten()
    nearest_indices = np.argsort(distances)[:top_k]
    return nearest_indices, distances[nearest_indices]

# Helper functions for embeddings
def encode_text(text):
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    query = tokenizer([text])
    query_embedding = F.normalize(clip_model.encode_text(query))
    return query_embedding

def encode_image(image_path):
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)
    image_embedding = F.normalize(clip_model.encode_image(img_tensor))
    return image_embedding

# Search function
def search(query_embedding, top_k=5):
    df['similarity'] = df['embedding'].apply(lambda x: F.cosine_similarity(torch.tensor(x), query_embedding.squeeze(0), dim=0))
    # Sort the DataFrame by similarity in descending order and get the top 5
    top5_images = df.sort_values('similarity', ascending=False).head(5)
    # Find the index of the image with the highest similarity
    images = []
    for index, row in top5_images.iterrows():
        impath = row['file_name']
        similarity = row['similarity']
        images.append((impath, similarity))
    return images

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        text_query = request.form.get("text_query", None)
        image_file = request.files.get("image_query", None)
        weight = float(request.form.get("weight", 0.5))
        use_pca = "use_pca" in request.form  # Checkbox for PCA usage
        k = int(request.form.get("pca_k", 50)) if use_pca else None  # Number of components

        # Compute embeddings for queries
        text_embedding = encode_text(text_query) if text_query else None
        image_embedding = encode_image(image_file) if image_file else None
        if use_pca:
            train_images, train_image_names = load_images(file_prefix, max_images=2000, target_size=(224, 224))
            pca = pca_train_images(train_images, k)
            transform_images, transform_image_names = load_images(file_prefix, max_images=10000, target_size=(224, 224))
            reduced_embeddings = pca_transform_images(transform_images, pca)
            img = load_image(image_file, target_size=(224,224))
            query_embedding = pca.transform([img])[0]
            top_indices, top_distances = nearest_neighbors(query_embedding, reduced_embeddings)
            res = []
            for i, idx in enumerate(top_indices):
                result_image = transform_image_names[idx]
                res.append((result_image, top_distances[i]))
            
            results = res
        else:

            # Combine embeddings
            if text_embedding is not None and image_embedding is not None:
                combined_embedding = weight * text_embedding + (1 - weight) * image_embedding
            elif text_embedding is not None:
                combined_embedding = text_embedding
            elif image_embedding is not None:
                combined_embedding = image_embedding
            else:
                combined_embedding = None

            # Perform search
            if combined_embedding is not None:
                results = search(combined_embedding, top_k=5)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
