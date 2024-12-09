# Install required packages  
!pip install flask flask-ngrok easyocr opencv-python-headless roboflow transformers torch pillow pyngrok  

# Import necessary libraries  
from flask import Flask, request, jsonify, render_template_string  
import easyocr  
import cv2  
from werkzeug.utils import secure_filename  
import os  
from roboflow import Roboflow  
import re  
import numpy as np  
from datetime import datetime  
from inference_sdk import InferenceHTTPClient  
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  
from PIL import Image  
import torch  
from pyngrok import ngrok  

# Initialize Flask app  
app = Flask(__name__)  

# Set your ngrok auth token  
ngrok.set_auth_token("2pxnSnSwqB70L03JMqSailJkz8B_2PaWAyzaPB5vSS8fSpg93")  # Replace with your actual ngrok auth token  

# Initialize the InferenceHTTPClient with your API URL and API key  
CLIENT = InferenceHTTPClient(  
    api_url="https://detect.roboflow.com",  
    api_key="YOUR_ROBOFLOW_API_KEY"  # Replace with your actual Roboflow API key  
)  

# Initialize Qwen model for brand detection  
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32)  
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")  

# Define the OCR model function  
def ocr_model(image_path):  
    # Call the Roboflow inference  
    result = CLIENT.infer(image_path, model_id="expiredatedetection/4")  

    # Print result to check its structure  
    print(result)  

    # Assuming result is a dictionary-like object and contains 'predictions'  
    predictions = result.get('predictions', [])  # Safely access predictions  

    # Process predictions  
    extracted_dates = []  
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR Reader  

    for prediction in predictions:  
        x_center = prediction["x"]  
        y_center = prediction["y"]  
        box_width = prediction["width"]  
        box_height = prediction["height"]  

        top_left_x = int(x_center - box_width / 2)  
        top_left_y = int(y_center - box_height / 2)  
        bottom_right_x = int(x_center + box_width / 2)  
        bottom_right_y = int(y_center + box_height / 2)  

        # Read and preprocess the cropped region  
        image = cv2.imread(image_path)  
        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]  
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)  
        _, processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

        # OCR on the processed image  
        result = reader.readtext(processed_image)  

        for bbox, text, confidence in result:  
            date_match_ymd = re.match(r"(\d{4})[.,]?\s*(\d{2})[.,]?\s*(\d{2})(?:/)?", text)  
            date_match_dmy = re.match(r"(\d{2})[.,]?\s*(\d{2})[.,]?\s*(\d{4})(?:/)?", text)  

            if date_match_ymd:  
                year = date_match_ymd.group(1)  
                month = date_match_ymd.group(2)  
                day = date_match_ymd.group(3)  
                extracted_dates.append({"year": year, "month": month, "day": day})  

            elif date_match_dmy:  
                day = date_match_dmy.group(1)  
                month = date_match_dmy.group(2)  
                year = date_match_dmy.group(3)  
                extracted_dates.append({"year": year, "month": month, "day": day})  

    return extracted_dates  

# Define the brand detection function  
def detect_brand(image_path):  
    image = Image.open(image_path)  
    messages = [  
        {  
            "role": "user",  
            "content": [  
                {"type": "image"},  
                {"type": "text", "text": "Extract brand name, pack size, number of the same product, and expiry date."},  
            ],  
        }  
    ]  
    # Prepare the text prompt  
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  
    inputs = processor(  
        text=[text_prompt],  
        images=[image],  
        padding=True,  
        return_tensors="pt"  
    )  

    # Generate output text  
    output_ids = model.generate(**inputs, max_new_tokens=256)  
    generated_ids = [  
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)  
    ]  

    # Decode the result  
    output_text = processor.batch_decode(  
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True  
    )  
    return output_text[0] if output_text else "No brand detected."  

# Define a route for the index page  
@app.route('/')  
def index():  
    return render_template_string(open('index.html').read())  

# Route to handle image upload and process the image  
@app.route('/process', methods=['POST'])  
def process_image():  
    if 'image' not in request.files:  
        return jsonify({"error": "No file part"}), 400  

    file = request.files['image']  

    if file.filename == '':  
        return jsonify({"error": "No selected file"}), 400  

    filename = secure_filename(file.filename)  
    file_path = os.path.join('uploads', filename)  
    file.save(file_path)  

    # Process the uploaded image using OCR model  
    extracted_data = ocr_model(file_path)  

    # Process the uploaded image using brand detection  
    brand_info = detect_brand(file_path)  

    # Prepare structured data for rendering in a table  
    table_data = []  
    current_date = datetime.now()  
    for i, date_info in enumerate(extracted_data):  
        expiry_date = datetime(  
            year=int(date_info["year"]),  
            month=int(date_info["month"]),  
            day=int(date_info["day"])  
        )  
        life_span_days = (expiry_date - current_date).days  
        expired = "No" if life_span_days > 0 else "Yes"  

        table_data.append({  
            "sl_no": i + 1,  
            "timestamp": current_date.isoformat(),  
            "brand": brand_info,  # Use detected brand info  
            "expiry_date": expiry_date.strftime("%d/%m/%Y"),  
            "expired": expired,  
            "expected_life_span_days": life_span_days if life_span_days > 0 else 0  
        })  

    return render_template_string(open('index.html').read(), table_data=table_data)  

# Create index.html file  
html_code = """  
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>OCR Model</title>  
    <style>  
        body {  
            font-family: Arial, sans-serif;  
            margin: 0;  
            padding: 0;  
            background-color: #f4f4f4;  
        }  
        .container {  
            max-width: 800px;  
            margin: 50px auto;  
            padding: 20px;  
            background-color: #fff;  
            border-radius: 10px;  
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  
        }  
        h1 {  
            text-align: center;  
        }  
        input[type="file"] {  
            display: block;  
            margin: 20px auto;  
        }  
        button {  
            display: block;  
            width: 100%;  
            padding: 10px;  
            font-size: 16px;  
            cursor: pointer;  
            background-color: #4CAF50;  
            color: white;  
            border: none;  
            border-radius: 5px;  
        }  
        table {  
            width: 100%;  
            margin-top: 20px;  
            border-collapse: collapse;  
        }  
        th, td {  
            border: 1px solid #ddd;  
            padding: 8px;  
            text-align: center;  
        }  
        th {  
            background-color: #4CAF50;  
            color: white;  
        }  
        .error {  
            color: red;  
            text-align: center;  
            margin-top: 20px;  
        }  
    </style>  
</head>  
<body>  
    <div class="container">  
        <h1>OCR Model - Date Extraction</h1>  
        <!-- Form to upload image -->  
        <form action="/process" method="POST" enctype="multipart/form-data">  
            <input type="file" name="image" accept="image/*" required>  
            <button type="submit">Process Image</button>  
        </form>  

        <!-- Display table if table_data exists -->  
        {% if table_data %}  
        <table>  
            <thead>  
                <tr>  
                    <th>Sl No</th>  
                    <th>Timestamp</th>  
                    <th>Brand</th>  
                    <th>Expiry Date</th>  
                    <th>Expired</th>  
                    <th>Expected Life Span (Days)</th>  
                </tr>  
            </thead>  
            <tbody>  
                {% for row in table_data %}  
                <tr>  
                    <td>{{ row.sl_no }}</td>  
                    <td>{{ row.timestamp }}</td>  
                    <td>{{ row.brand }}</td>  
                    <td>{{ row.expiry_date }}</td>  
                    <td>{{ row.expired }}</td>  
                    <td>{{ row.expected_life_span_days }}</td>  
                </tr>  
                {% endfor %}  
            </tbody>  
        </table>  
        {% else %}  
        <p class="error">No data extracted yet. Please upload an image.</p>  
        {% endif %}  
    </div>  
</body>  
</html>  
"""  

# Save the HTML code to index.html  
with open('index.html', 'w') as f:  
    f.write(html_code)  

# Set up a tunnel to the Flask app  
public_url = ngrok.connect(5000)  
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:5000\"".format(public_url))  

# Run the Flask app  
app.run(port=5000)
