# Install required packages  
!pip install flask flask-ngrok easyocr opencv-python-headless roboflow transformers torch pillow pyngrok  

import os
import cv2
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import streamlit as st
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
from pyngrok import ngrok

# Define a multipage app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["OCR", "Freshness Detection", "Expiry Date Verification", "Processed Products"])

# Global variables to store processed product details
if 'processed_products' not in st.session_state:
    st.session_state.processed_products = []

if 'freshness_products' not in st.session_state:
    st.session_state.freshness_products = []

if 'expiry_products' not in st.session_state:
    st.session_state.expiry_products = []

# Load Qwen model for OCR and Expiry Date
@st.cache_resource
def load_qwen_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor

model, processor = load_qwen_model()

def preprocess_image(image):
    image_np = np.array(image)
    resized_image = cv2.resize(image_np, (640, 480))
    return Image.fromarray(resized_image)

# OCR Page
if page == "OCR":
    st.title("OCR: Extract Product Details")
    uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Extract Product Details"):
            with st.spinner("Processing the image..."):
                # Extract details
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "extract brand name, pack size, and expiry date"}
                        ]
                    }
                ]
                text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
                output_ids = model.generate(**inputs, max_new_tokens=512)
                output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                extracted_details = output_text[0]

                st.success("Product details extracted successfully!")
                st.markdown(f"*Extracted Text:* {extracted_details}")

                # Parse details and update table
                brand_name = extracted_details.split(',')[0].split(':')[-1].strip()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.processed_products.append({
                    "Timestamp": timestamp,
                    "Extracted Brand Details": extracted_details
                })

# Freshness Detection Page
elif page == "Freshness Detection":
    st.title("Freshness Detection: Object Counter")
    
    @st.cache_resource
    def load_freshness_model():
        from ultralytics import YOLO
        return YOLO("best.pt")  # Provide the correct path to your model file

    model = load_freshness_model()

    uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        temp_file = "temp_image.jpg"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Run Detection"):
            with st.spinner("Processing the image..."):
                results = model.predict(source=temp_file, conf=0.3, imgsz=640)
                image = cv2.imread(temp_file)
                counts = Counter()

                # Process detections
                for result in results:
                    for box in result.boxes.data.tolist():
                        x1, y1, x2, y2, confidence, class_id = box
                        class_name = model.names[int(class_id)]
                        counts[class_name] += 1
                        freshness = "Fresh" if "fresh" in class_name.lower() else "Rotten"
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{class_name} {confidence:.2f} ({freshness})"
                        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                st.image(image, caption="Processed Image with Detections", use_column_width=True)
                st.write("### Object Counts:")
                for class_name, count in counts.items():
                    st.write(f"- {class_name}: {count}")

# Expiry Date Verification Page
elif page == "Expiry Date Verification":
    st.title("Expiry Date Verification with Qwen")
    uploaded_file = st.file_uploader("Upload a product image for expiry date verification", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Verify Expiry Date"):
            with st.spinner("Processing the image..."):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "extract expiry date from this product image"}
                        ]
                    }
                ]
                text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
                output_ids = model.generate(**inputs, max_new_tokens=512)
                output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                extracted_expiry_date = output_text[0]

                st.success(f"Extracted Expiry Date: {extracted_expiry_date}")
                validation_status = "Valid" if extracted_expiry_date else "Invalid"
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.expiry_products.append({
                    "Timestamp": timestamp,
                    "Expiry Date Extracted": extracted_expiry_date or "NA",
                    "Status": validation_status
                })

# Processed Products Page
elif page == "Processed Products":
    st.title("Processed Products and Relations")
    
    if st.session_state.processed_products:
        df_products = pd.DataFrame(st.session_state.processed_products)
        st.write("### Processed Products (OCR)")
        st.dataframe(df_products, use_container_width=True)
    else:
        st.write("No OCR data available.")

