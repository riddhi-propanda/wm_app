import base64
import zipfile
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from flask import Flask, request, jsonify, send_file
import os
import cv2
import numpy as np
from PIL import Image
import io
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text, property_name, location):

    ### LLama2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens': 256,
                              'temperature': 0.01})

    ## Prompt Template
    template = """
        Write an SEO-friendly property listing for {property_name} located in {location}.
        The property features include {input_text}.
    """

    prompt = PromptTemplate(input_variables=["property_name", "location", "input_text"],
                          template=template)

    ## Generate the response from the LLama 2 model
    response = llm(prompt.format(property_name=property_name, location=location, input_text=input_text))
    print(response)
    return response

# Function to process the image
def process_images(files, upload_dir):
    processed_image_paths = []
    for file in files:
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Unable to process image. Please ensure it is a valid image file.'}), 400

        resized_image = cv2.resize(image, (1024, 768))

        mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)
        height, width = resized_image.shape[:2]
        rect_width = width // 4
        rect_height = height // 23
        start_x = (width - rect_width) // 2
        start_y = (height - rect_height) // 2
        end_x = start_x + rect_width
        end_y = start_y + rect_height
        mask[start_y:end_y, start_x:end_x] = 255

        dst = cv2.inpaint(resized_image, mask, 3, cv2.INPAINT_TELEA)

        watermark = np.zeros_like(resized_image, dtype=np.uint8)
        text = "PROPERTYPANDA.COM"
        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = resized_image.shape[1] / 350

        textsize = cv2.getTextSize(text, font, font_scale, 2)[0]
        textX = (resized_image.shape[1] - textsize[0]) // 12
        textY = (resized_image.shape[0] + textsize[1]) // 2

        overlay = dst.copy()
        cv2.putText(overlay, text, (textX, textY), font, font_scale, (250, 250, 250), 5, cv2.LINE_AA)
        alpha = 0.5

        cv2.addWeighted(overlay, alpha, resized_image, 1 - alpha, 0, resized_image)

        filename = secure_filename(file.name)
        # print(filename)
        ext = filename.split(".")[-1]
        new_filename = filename.replace(ext, "webp")
        processed_image_path = os.path.join(upload_dir, new_filename)
        cv2.imwrite(processed_image_path, resized_image)
        processed_image_paths.append(processed_image_path)

    return processed_image_paths



def get_binary_file_downloader_html(bin_file, label='Download Zip File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{label}</a>'
    return href

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
upload_dir = os.path.join(os.path.dirname(script_dir), 'storage', 'app', 'public', 'store_tmp_images')
os.makedirs(upload_dir, exist_ok=True)

@app.route('/api/watermark', methods=['POST'])
def watermark_route():
    # Route to handle watermarking request
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files[]')

    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    allowed_extensions = {'jpg', 'jpeg', 'png'}
    for file in files:
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format for file: {file.filename}. Supported formats are: jpg, jpeg, png'}), 400

    processed_image_paths = process_images(files, upload_dir)
    return jsonify({'processed_image_paths': processed_image_paths})


st.set_page_config(page_title="Generate Property Listings",
                    page_icon='🏠',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Property Listings 🏠")

tabs = ["Property Listing Generation", "Image Watermarking", "PDF Modification"]
selected_tab = st.radio("Select Task", tabs)

if selected_tab == "Property Listing Generation":
    input_text = st.text_area("Enter the Property Features", height=100)
    property_name = st.text_input("Enter the Property Name")
    location = st.text_input("Enter the Location")

    submit = st.button("Generate")

    ## Final response
    if submit:
        st.write(getLLamaresponse(input_text, property_name, location))
elif selected_tab == "Image Watermarking":
    st.write("Upload images to add watermark.")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write("Uploaded Images:")
        processed_image_paths = process_images(uploaded_files, upload_dir)
        for uploaded_file, processed_image_path in zip(uploaded_files, processed_image_paths):
            st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
            st.image(processed_image_path, caption="Watermarked Image", use_column_width=True)

        if st.button("Download All Images"):
            zip_filename = 'watermarked_images.zip'
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for image_path in processed_image_paths:
                    zipf.write(image_path, os.path.basename(image_path))

            st.markdown(get_binary_file_downloader_html(zip_filename, 'Images Zip'), unsafe_allow_html=True)
elif selected_tab == "PDF Modification":
    st.write("Upload a PDF file for modification.")
    uploaded_pdf = st.file_uploader("Choose a PDF file...", type=["pdf"])

    if uploaded_pdf:
        st.write("PDF Uploaded:", uploaded_pdf.name)
        if st.button("Process PDF"):
            # Send PDF to Flask API for processing
            response = requests.post('http://localhost:5000/process_pdf', files={'file': uploaded_pdf})
            if response.status_code == 200:
                st.success("PDF processed successfully.")
                st.markdown(get_binary_file_downloader_html(response.content, 'Processed PDF'), unsafe_allow_html=True)
            else:
                st.error("Error processing PDF. Please try again.")


