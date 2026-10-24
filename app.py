import streamlit as st
import requests

st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="centered")

st.title("🎨 Clipdrop Image Generator")
st.write("Generate amazing images using Clipdrop API directly from this app!")

prompt = st.text_area("Enter your prompt:", placeholder="A futuristic city at sunset...")

if st.button("Generate Image", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Generating image... please wait..."):
            # Your backend / API call logic
            st.info("Triggering image generation for: " + prompt)
            st.success("App connected successfully to Streamlit Cloud!")
