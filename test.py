# Fallback version without pgvector extension
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import psycopg2
import os
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    device = 'cpu'
    print("Using CPU")

# Load the CLIP model
print("Loading CLIP model...")
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
print("Model loaded successfully!")

# connecting to local PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="Facial-Embbedings",
    user="postgres",
    password="1234",
    port="5432"
)

# loading the face image path into file_name variable (using raw string)
file_name = r"D:\SAKIB\NSU\COURSES\CSE 499B\FIRST BENCHMARK\CODE\FACIAL RECOGNITION\test_image\S_2702_5.jpg"

try:
    # opening the image
    img = Image.open(file_name)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # calculating the embeddings using CLIP
    query_embedding = model.encode(img)
    
    # Get all embeddings from database and calculate similarity in Python
    cur = conn.cursor()
    cur.execute("SELECT filename, embedding FROM pictures;")
    rows = cur.fetchall()
    
    similarities = []
    
    for filename, db_embedding in rows:
        # Convert database embedding to numpy array
        db_embedding_array = np.array(db_embedding)
        
        # Calculate cosine similarity
        dot_product = np.dot(query_embedding, db_embedding_array)
        norm_query = np.linalg.norm(query_embedding)
        norm_db = np.linalg.norm(db_embedding_array)
        similarity = dot_product / (norm_query * norm_db)
        
        similarities.append((filename, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFound {len(similarities)} images in database:")
    print("-" * 50)
    
    # Show top 5 most similar
    for i, (filename, similarity) in enumerate(similarities[:5], 1):
        print(f"{i}. Most similar image: {filename} (similarity: {similarity:.4f})")
        
        # Display image path for reference
        image_path = os.path.join("Image_Albums", filename)
        if os.path.exists(image_path):
            print(f"   Path: {image_path}")
        else:
            print(f"   Warning: Image file not found at {image_path}")
        print()
    
    cur.close()

except FileNotFoundError:
    print(f"Error: Query image '{file_name}' not found!")
    print("Please make sure the file exists in the current directory.")

except Exception as e:
    print(f"Error: {str(e)}")

finally:
    # Close the connection
    conn.close()
    print("Database connection closed.")