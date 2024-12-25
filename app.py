import streamlit as st
from openai import OpenAI
import os
import base64
from PIL import Image
from io import BytesIO

# Set the API key and model name
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Function to encode image as base64
def encode_image(image: Image.Image):
    # Convert PIL Image to byte array for base64 encoding
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")

# Function to save and process the uploaded image
def save_and_process_image(image: Image.Image, user_question: str):
    # Save the uploaded image
    image_path = "uploaded_image.png"
    image.save(image_path)
    st.write(f"Image saved at: {image_path}")

    # Pass the saved image to the lvlm function
    result = lvlm(image, user_question)
    return result

# Function to communicate with the LVLM
def lvlm(image: Image.Image, user_question: str):
    # Encode the image to base64
    base64_image = encode_image(image)
    
    # Make the request to OpenAI API
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "system", 
            "content": "You are a helpful assistant that responds in Vietnamese. Help me solve my math homework step-by-step. Note that put the formulas in a pair of $$."
        }, {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }],
        temperature=0.0,
    )

    return response.choices[0].message.content

# Streamlit UI
def main():
    st.title("Image Upload and LVLM Integration")

    # User inputs a custom question
    user_question = st.text_input("Enter your question:")

    # Allow user to upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image with lvlm
        if st.button("Analyze Uploaded Image"):
            with st.spinner("Processing..."):
                result = save_and_process_image(image, user_question)
                st.write(result)

    else:
        st.write("Please upload an image to begin analysis.")

if __name__ == "__main__":
    main()
