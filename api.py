from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import clip
from PIL import Image
from qdrant_client import QdrantClient
import io
import os

app = FastAPI(title="Semantic Image Search API")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "unsplash_25kv2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP model on {DEVICE}...")
model, preprocess = clip.load("ViT-B/16", device=DEVICE)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    path: str
    filename: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

def encode_text(text: str) -> List[float]:
    with torch.no_grad():
        text_inputs = clip.tokenize([text]).to(DEVICE)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()

def encode_image(image: Image.Image) -> List[float]:
    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()

def query_qdrant(vector: List[float], top_k: int) -> List[SearchResult]:
    try:
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k
        ).points
        
        results = []
        for point in search_result:
            results.append(SearchResult(
                path=point.payload.get('path', ''),
                filename=point.payload.get('filename', ''),
                score=point.score
            ))
        return results
    except Exception as e:
        print(f"Qdrant Error: {e}")
        return []

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": DEVICE, "qdrant_host": QDRANT_HOST}

@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(request: SearchRequest):
    print(f"Processing text query: {request.query}")
    vector = encode_text(request.query)
    results = query_qdrant(vector, request.top_k)
    return SearchResponse(results=results)

@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(file: UploadFile = File(...), top_k: int = 5):
    print(f"Processing image query: {file.filename}")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        vector = encode_image(image)
        results = query_qdrant(vector, top_k)
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
