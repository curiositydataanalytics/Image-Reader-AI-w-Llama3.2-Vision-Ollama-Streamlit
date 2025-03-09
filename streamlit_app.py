# Data manipulation
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd

# Database and file handling
import os

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk

import ollama
from PIL import Image
import base64
import io
import time

path_cda = '\\CuriosityDataAnalytics'
path_wd = path_cda + '\\wd'
path_data = path_wd + '\\data'

# App config
#----------------------------------------------------------------------------------------------------------------------------------#
# Page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .element-container {
        margin-top: -2px;
        margin-bottom: -2px;
        margin-left: -2px;
        margin-right: -2px;
    }
    img[data-testid="stLogo"] {
                height: 6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title("Image Reader AI w/ Llama3.2-Vision")
st.divider()

with st.sidebar:
    st.logo(path_cda + '\\logo.png', size='large')
    st.empty()

#
#

if "describe" not in st.session_state:
    st.session_state['describe'] = None
if "extract" not in st.session_state:
    st.session_state['extract'] = None
if "summarize" not in st.session_state:
    st.session_state['summarize'] = None


cols = st.columns((0.4,0.6))

with cols[0]:
    st.subheader('Input')
    image_input = st.file_uploader(' ', type=["png", "jpg", "jpeg"])

    if image_input:
        st.image(image_input)
        image = Image.open(image_input)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

with cols[1]:
    if image_input:
        
        subcols = st.columns(5)
        subcols[0].subheader('Output')
        with st.spinner('Processing...'):

            describe = ollama.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': 'Describe the content of this image in detail, including objects, scenes, and any visible context. Provide a concise and well-organized description.',
                    'images': [image_base64]
                }]
            )

            with st.expander('Image Description'):
                st.markdown(
                    f"""
                    <div style="color: gold; padding: 10px; line-height: 1.6;">
                        {describe['message'].content}
                    """,
                    unsafe_allow_html=True)
                
            extract = ollama.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': 'Extract all text from this image accurately, preserving line breaks and formatting where possible. Output the text as plain text without any explanations or comments.',
                    'images': [image_base64]
                }]
            )

            with st.expander('Text Extraction'):
                st.markdown(
                    f"""
                    <div style="color: gold; padding: 10px; line-height: 1.6;">
                        {extract['message'].content}
                    """,
                    unsafe_allow_html=True)
                
            summarize = ollama.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': 'Extract the text from this image and summarize its main points. Provide a brief summary in 3-4 bullet points.',
                    'images': [image_base64]
                }]
            )

            with st.expander('Text Summarization'):
                st.markdown(
                    f"""
                    <div style="color: gold; padding: 10px; line-height: 1.6;">
                        {summarize['message'].content}
                    """,
                    unsafe_allow_html=True)


        
