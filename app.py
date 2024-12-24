import streamlit as st
from openai import OpenAI
import os
import base64
from PIL import Image
from io import BytesIO

# Set the API key and model name
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", " "))

# Function to encode image as base64
def encode_image(image: Image.Image):
    # Convert PIL Image to byte array for base64 encoding
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")

# Function to communicate with OpenAI API
def lvlm(image: Image.Image, user_question: str):
    # Encode the image to base64
    base64_image = encode_image(image)

    # Make the request to OpenAI API
    response = client.chat.completions.create(
        model=MODEL,  # Replace with the correct model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please help me with the following question: {user_question}"},
            {"role": "user", "content": f"Here is the image you need to analyze: data:image/png;base64,{base64_image}"}
        ],
        temperature=0.0,
    )

    return response.choices[0].message['content']

if __name__ == '__main__':
    # Streamlit app layout
    st.title("Capture Image from Webcam")

    # Create a camera input widget to capture an image
    camera_input = st.camera_input("Take a picture")

    # Check if an image was captured
    if camera_input is not None:
        # Convert the captured image into a PIL image
        captured_image = Image.open(camera_input)

        # Display the captured image
        st.image(captured_image, caption="Captured Image", use_column_width=True)

        # Optionally, save the captured image
        captured_image.save("captured_image.jpg")
        st.write("Image saved as 'captured_image.jpg'")

        if captured_image is not None:
            # User inputs a custom question
            user_question = st.text_input("Enter your question:")

            # Get result from OpenAI
            if user_question:
                result = lvlm(captured_image, user_question)  # Pass PIL Image object
                st.write(result)


