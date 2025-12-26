import os
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from tqdm import tqdm

IMAGE_FOLDER = r"C:\Users\rohan\Downloads\unsplash\data\train"
COLLECTION_NAME = "unsplash_25kv2"
BATCH_SIZE = 128
VECTOR_SIZE = 512

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.image_paths = []
        self.transform = transform
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(folder_path, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = self.transform(Image.open(path))
            return image, path
        except Exception as e:
            return torch.zeros((3, 224, 224)), "ERROR"

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')
    
    model, preprocess = clip.load("ViT-B/16", device=device)

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created...")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists...")

    print(f"Indexing the content of the folder: {IMAGE_FOLDER}...")
    dataset = ImageDataset(IMAGE_FOLDER, preprocess)
    
    if len(dataset) == 0:
        print("ERROR: No images found!")
        exit()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    print(f"Starting indexing of {len(dataset)} images...")
    
    idx_counter = 0
    
    with torch.no_grad():
        for batch_images, batch_paths in tqdm(dataloader, desc="Processing Batches"):
            batch_images = batch_images.to(device)
            
            batch_features = model.encode_image(batch_images)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            
            points_buffer = []
            
            for vector, path in zip(batch_features, batch_paths):
                if path == "ERROR": continue
                
                point = PointStruct(
                    id=idx_counter,
                    vector=vector.cpu().tolist(),
                    payload={"filename": os.path.basename(path), "path": path} 
                )
                points_buffer.append(point)
                idx_counter += 1
            
            if points_buffer:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_buffer
                )
    
    print(f"Successfully indexed {idx_counter} images.")