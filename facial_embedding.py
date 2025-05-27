# importing the required libraries
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import psycopg2
import os
import torch
import cv2

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    device = 'cpu'
    print("Using CPU")

# Load the CLIP model for image embeddings (this will use GPU automatically if available)
print("Loading CLIP model...")
model = SentenceTransformer('clip-ViT-B-32', device=device)
print("Model loaded successfully!")

# connecting to local PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="Facial-Embbedings",
    user="postgres",
    password="1234",
    port="5432"
)

# Make sure the pictures table exists
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS pictures (
        filename VARCHAR(255) PRIMARY KEY,
        embedding FLOAT8[]
    );
""")
conn.commit()

# Process images in the Image_Albums directory
if os.path.exists("Image_Albums"):
    image_files = [f for f in os.listdir("Image_Albums") 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    print(f"Found {len(image_files)} image files to process...")
    
    for i, filename in enumerate(image_files, 1):
        try:
            # opening the image
            image_path = os.path.join("Image_Albums", filename)
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # calculating the embeddings using CLIP
            embedding = model.encode(img)
            
            # inserting into database with conflict handling
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO pictures (filename, embedding) 
                VALUES (%s, %s) 
                ON CONFLICT (filename) DO UPDATE SET embedding = EXCLUDED.embedding
            """, (filename, embedding.tolist()))
            
            print(f"Processed ({i}/{len(image_files)}): {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    conn.commit()
    print("All images processed successfully!")
else:
    print("Directory 'Image_Albums' not found!")

# Close the connection
conn.close()