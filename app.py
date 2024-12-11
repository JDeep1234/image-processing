import streamlit as st  
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor  
from PIL import Image  
from datetime import datetime  
import re  

# Load model and processor  
@st.cache_resource  
def load_model():  
    model = Qwen2VLForConditionalGeneration.from_pretrained(  
        "Qwen/Qwen2-VL-2B-Instruct",  
        torch_dtype="auto",  
        device_map="auto",  
    )  
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")  
    return model, processor  

model, processor = load_model()  

# Function to analyze the uploaded image  
def analyze_image(image):  
    messages = [  
        {  
            "role": "user",  
            "content": [  
                {"type": "image"},  
                {"type": "text", "text": "Read brand details, pack size, brand name, expiry date, and MRP"}  
            ]  
        }  
    ]  

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  

    inputs = processor(  
        text=[text_prompt],  
        images=[image],  
        padding=True,  
        return_tensors="pt"  
    )  

    output_ids = model.generate(**inputs, max_new_tokens=1024)  

    generated_ids = [  
        output_ids[len(input_ids):]  
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)  
    ]  

    output_text = processor.batch_decode(  
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True  
    )[0]  

    return output_text  

# Function to extract expiry date and MRP from output text  
def extract_details(output_text):  
    date_patterns = [  
        r'\b(\d{2}/\d{2}/\d{4})\b',  
        r'\b(\d{2}-\d{2}-\d{4})\b',  
        r'\b(\d{2}/\d{2}/\d{2})\b',  
        r'\b(\d{2}-\d{2}-\d{2})\b',  
        r'\b(\d{2} \w+ \d{4})\b',  
        r'\b(\d{2} \d{2} \d{4})\b'  
    ]  

    expiry_date = None  
    for pattern in date_patterns:  
        match = re.findall(pattern, output_text)  
        if match:  
            expiry_date = match[0]  
            break  

    if expiry_date:  
        try:  
            if " " in expiry_date:  
                expiry_date = datetime.strptime(expiry_date, "%d %m %Y")  
            elif "/" in expiry_date or "-" in expiry_date:  
                expiry_date = datetime.strptime(expiry_date, "%d/%m/%Y")  

            current_date = datetime.now()  

            return expiry_date.strftime('%d/%m/%Y'), expiry_date < current_date  
        except ValueError:  
            return "Invalid date format", False  
    else:  
        return "No valid expiry date found", False  

# Streamlit app layout  
st.title("Product Details Extractor")  
st.write("Upload an image of the product to extract details.")  

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])  

if uploaded_file is not None:  
    image = Image.open(uploaded_file)  
    st.image(image, caption='Uploaded Image', use_container_width=True)  

    if st.button("Analyze"):  
        with st.spinner("Analyzing..."):  
            output_text = analyze_image(image)  
            expiry_date, is_expired = extract_details(output_text)  

            st.subheader("Extracted Details")  
            st.write("Output Text:")  
            st.write(output_text)  

            # Display the results in a table  
            st.table({  
                "Expiry Date": [expiry_date],  
                "Status": ["Expired" if is_expired else "Valid"]  
            })
