import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import os
import replicate

# Page Config
st.set_page_config(page_title="AI Barber Shop Try-On", layout="centered")

st.title("✂️ AI Barber Shop Virtual Try-On")
st.write("Snap a photo, brush over your hair, choose a style, and see the magic!")

# Ensure API Key is set
if not os.environ.get("REPLICATE_API_TOKEN"):
    st.error("Please set your REPLICATE_API_TOKEN environment variable to run the AI model!")
    st.stop()

# 1. Image Input Method
option = st.radio("Choose photo source:", ("Take Live Selfie", "Upload a Photo"))
src_image = None

if option == "Take Live Selfie":
    camera_photo = st.camera_input("Smile for the camera!")
    if camera_photo is not None:
        src_image = Image.open(camera_photo)
else:
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        src_image = Image.open(uploaded_file)

# 2. Style Selection & Processing
if src_image:
    # Resize image to a manageable size for processing & displaying
    src_image.thumbnail((800, 800))
    width, height = src_image.size

    st.subheader("🖌️ Step 1: Paint over your hair")
    st.info("Use the brush to cover all the hair you want to replace. Keep your face untouched!")

    # Set up drawing canvas
    stroke_width = st.slider("Brush size:", 10, 100, 35)
    
    # Show drawable canvas on top of user photo
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # Inpainting mask uses white for replacement
        stroke_width=stroke_width,
        stroke_color="rgba(255, 255, 255, 1)",
        background_image=src_image,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="freedraw",
        key="canvas",
    )

    # 3. Choose/Write Style
    st.subheader("💈 Step 2: Choose your style")
    
    preset_styles = {
        "Classic Pompadour": "A neat classic pompadour style haircut, professional barber, photorealistic",
        "Messy Textured Crop": "Modern textured messy crop haircut, stylish, matte finish, photorealistic",
        "Buzz Cut": "Clean buzz cut haircut, short military style, faded sides, professional barber, photorealistic",
        "Buzz Cut Fade": "Buzz cut with high skin fade, crisp line-up, modern barber look, photorealistic",
        "Long Waves": "Shoulder-length wavy hair, casual and stylish, photorealistic"
    }
    
    style_type = st.radio("Style option:", ("Preset Styles", "Custom Description"))
    
    prompt = ""
    if style_type == "Preset Styles":
        preset_choice = st.selectbox("Select a preset:", list(preset_styles.keys()))
        prompt = preset_styles[preset_choice]
    else:
        custom_input = st.text_input(
            "Describe your dream hairstyle:", 
            placeholder="e.g., A stylish silver undercut fade with textured spikes on top"
        )
        if custom_input:
            prompt = f"{custom_input}, highly detailed, professional barber haircut, photorealistic"

    # 4. Generate Button
    if st.button("✨ Apply Haircut!", type="primary"):
        if canvas_result.image_data is not None and prompt:
            with st.spinner("Our AI Barber is working on your hair..."):
                try:
                    # Convert canvas drawing into a PIL Mask Image
                    # Canvas returns RGBA, we need to extract the white brush strokes on a black background
                    mask_data = canvas_result.image_data
                    mask_image = Image.fromarray(mask_data).convert("L")
                    
                    # Convert original image and mask to bytes
                    img_byte_arr = io.BytesIO()
                    src_image.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()

                    mask_byte_arr = io.BytesIO()
                    mask_image.save(mask_byte_arr, format='PNG')
                    mask_bytes = mask_byte_arr.getvalue()

                    # Trigger the FLUX Fill model on Replicate
                    output = replicate.run(
                        "black-forest-labs/flux-fill-pro",
                        input={
                            "image": img_bytes,
                            "mask": mask_bytes,
                            "prompt": prompt,
                            "steps": 30,
                            "guidance": 50,
                            "output_format": "jpg"
                        }
                    )
                    
                    # Output is usually a URL to the final image
                    if output:
                        st.subheader("🔥 Your New Look!")
                        st.image(output[0] if isinstance(output, list) else output, use_container_width=True)
                        st.balloons()
                    else:
                        st.error("Failed to generate your hairstyle. Please try again.")

                except Exception as e:
                    st.error(f"Something went wrong with the AI: {e}")
        else:
            st.warning("Please brush your hair on the image and choose/write a style first!")
