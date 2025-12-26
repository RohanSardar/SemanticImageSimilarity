import streamlit as st
import requests
from PIL import Image
import os
import io

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Semantic Image Search", layout="wide", page_icon="🔍")

st.title("🔍 Semantic Image Search")

device = requests.get(f"{API_URL}/health")
st.badge("Using OpenAI's CLIP (Contrastive Language-Image Pre-training) architecture", icon=":material/check:", color="yellow")
st.markdown(f":orange-badge[**Backend:** Qdrant via Docker] :gray-badge[**Model:** CLIP ViT-B/16] :green-badge[**Device:** {device.json()['device']}]")

def search_by_text(query, top_k=5):
    try:
        response = requests.post(f"{API_URL}/search/text", json={"query": query, "top_k": top_k}, timeout=30)
        if response.status_code == 200:
            return response.json()['results']
        else:
            st.error(f"Backend Error: {response.text}")
            return []
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to backend at {API_URL}. Is it running?")
        return []

def search_by_image(image, top_k=5):
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        files = {'file': ('query_image.jpg', img_byte_arr, 'image/jpeg')}
        params = {'top_k': top_k}
        
        response = requests.post(f"{API_URL}/search/image", files=files, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()['results']
        else:
            st.error(f"Backend Error: {response.text}")
            return []
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to backend at {API_URL}. Is it running?")
        return []

def display_results(results):
    if not results:
        st.warning("No results found.")
        return

    st.subheader("Search Results")
    col_size = 2
    cols = st.columns(col_size)
    
    IMAGES_DIR = os.getenv("IMAGES_DIR", "data/images")
    if not os.path.exists(IMAGES_DIR) and not os.path.isabs(IMAGES_DIR):
         if os.path.exists(os.path.join(os.getcwd(), IMAGES_DIR)):
             IMAGES_DIR = os.path.join(os.getcwd(), IMAGES_DIR)

    if not os.path.exists(IMAGES_DIR):
        st.sidebar.warning(f"Image directory not found: {IMAGES_DIR}")
        st.sidebar.info("Set IMAGES_DIR environment variable to your image folder.")

    for idx, result in enumerate(results):
        filename = result.get('filename')
        score = result.get('score')
        
        if filename:
            path = os.path.join(IMAGES_DIR, filename)
        else:
            path = result.get('path') 

        with cols[idx%col_size]:
            if os.path.exists(path):
                st.image(path, width='content')
                st.caption(f"**Cosine Similarity: {score:.3f}**")
            else:
                st.warning(f"Image not found locally: {filename}")
                st.caption(f"looked in: {path}")

with st.sidebar:
    st.header("⚙️ Search Options")
    search_mode = st.radio("Search Mode", ["📝 Text Search", "🖼️ Image Search"])
    top_k = st.slider("Results count", min_value=1, max_value=10, value=5)

if search_mode == "📝 Text Search":
    st.subheader("Search by Description")
    query_text = st.text_input("Enter your search query:", placeholder="A dog in a field")
    
    if st.button("Search") and query_text:
        with st.spinner("Searching via API..."):
            results = search_by_text(query_text, top_k)
            display_results(results)

elif search_mode == "🖼️ Image Search":
    st.subheader("Search by Image Reference")
    uploaded_file = st.file_uploader("Upload an image to find similar ones", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)
        results = None
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(query_image, caption="Query Image", width=500)
        
        with col2:
            st.write("Find similar images")
            if st.button("Search"):
                with st.spinner("Uploading & Searching..."):
                    results = search_by_image(query_image, top_k)
        
        if results is not None:
            display_results(results)
