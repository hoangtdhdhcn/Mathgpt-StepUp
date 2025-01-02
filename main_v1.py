import streamlit as st
from openai import OpenAI
import os
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import base64

# Set the OpenAI API key
MODEL = "gpt-4-turbo"
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

    # Pass the saved image to the LVLM function
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
            "content": "You are a helpful assistant that responds in Vietnamese. Help me solve my math homework step-by-step. Note that put the formulas in a pair of $."
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

# Function to load and display the image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Streamlit App UI
def main():
    st.title("Image Upload, Crop, and LVLM Integration")

    # User inputs a custom question
    user_question = st.text_input("Enter your question:")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the image
        img = load_image(uploaded_file)

        # Resize the image based on the column width
        img_width, img_height = img.size
        max_width = st.sidebar.slider("Set Max Width for Image", min_value=100, max_value=1200, value=800)

        # Resize image if it's too large
        if img_width > max_width:
            img = img.resize((max_width, int(img_height * (max_width / img_width))))

        # Convert the image to RGBA for canvas compatibility
        img = img.convert("RGBA")

        # Canvas setup for cropping
        st.subheader("Draw the crop area (Rectangle) on the image")

        # Initialize session state variables
        if "shapes" not in st.session_state:
            st.session_state["shapes"] = []

        # Create the canvas with the image as the background
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparent background for drawing
            stroke_width=2,
            stroke_color="red",
            background_color="white",
            background_image=img,  # Set uploaded image as the background
            width=img.width,
            height=img.height,
            drawing_mode="rect",  # Set drawing mode to rectangle
            key="canvas",  # Unique key for the canvas
            initial_drawing=None,
            display_toolbar=True,
            update_streamlit=True,  # Let the canvas update Streamlit automatically
        )

        # Check for drawing data and process
        if canvas_result.json_data is not None:
            # Extract shapes drawn on the canvas
            shapes = canvas_result.json_data["objects"]
            if len(shapes) > 0:
                # Get the last shape drawn (rectangle)
                shape = shapes[-1]
                left = shape["left"]
                top = shape["top"]
                right = left + shape["width"]
                bottom = top + shape["height"]

                # Crop the image using the coordinates from the rectangle
                cropped_img = img.crop((left, top, right, bottom))

                # Display the cropped image
                st.image(cropped_img, caption="Cropped Image", use_column_width=True)

                # Save cropped image to the current directory
                cropped_img_path = "cropped_image.png"
                cropped_img.save(cropped_img_path)
                st.success(f"Image saved as {cropped_img_path}")

                # Provide a download link
                with open(cropped_img_path, "rb") as file:
                    st.download_button(
                        label="Download Cropped Image",
                        data=file,
                        file_name="cropped_image.png",
                        mime="image/png"
                    )

                # Now use the cropped image for LVLM processing
                if st.button("Analyze Cropped Image"):
                    with st.spinner("Processing..."):
                        result = save_and_process_image(cropped_img, user_question)
                        st.write(result)

    else:
        st.write("Please upload an image to begin analysis.")

if __name__ == "__main__":
    main()
