import os
import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(device)


embedding_model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)
embedding_model.to(device)


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to generate captions for images
def generate_captions(image_paths):
    captions = {}
    images = []
    
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        images.append(image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    for image_path, caption in zip(image_paths, preds):
        captions[os.path.basename(image_path)] = caption
    
    return captions


# Function to compute embeddings for captions
def get_caption_embedding(caption):
    inputs = embedding_tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Function to find the most similar caption
def find_most_similar_caption(user_caption, generated_captions):
    user_embedding = get_caption_embedding(user_caption)
    similarities = {}

    
    for image_name, caption in generated_captions.items():
        caption_embedding = get_caption_embedding(caption)
        similarity_score = cosine_similarity(user_embedding, caption_embedding)[0][0]
        similarities[image_name] = similarity_score

    
    most_similar_image = max(similarities, key=similarities.get)
    
    return most_similar_image, similarities[most_similar_image]


st.title("SnapMatch")

user_caption = st.text_input("Enter a caption for comparison:")
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

if st.button("Find Most Similar Image") and uploaded_files and user_caption:
    image_files = [file.name for file in uploaded_files]
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())


    st.write("Generating captions for images...")
    generated_captions = generate_captions(image_files)


    most_similar_image, similarity_score = find_most_similar_caption(user_caption, generated_captions)

    st.write(f"Most similar image: {most_similar_image} with similarity score: {similarity_score}")
    st.image(most_similar_image)


    for file in image_files:
        os.remove(file)

