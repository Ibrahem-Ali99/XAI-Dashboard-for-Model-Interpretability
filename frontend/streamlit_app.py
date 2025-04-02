import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from io import StringIO

FLASK_URL = "http://localhost:5000"

FILE_TYPES = {
    "CSV": {"type": ["csv"], "reader": pd.read_csv},
    "Excel": {"type": ["xlsx", "xls"], "reader": pd.read_excel},
    "JSON": {"type": ["json"], "reader": pd.read_json},
    "Pickle (Python)": {"type": ["pkl"], "reader": pd.read_pickle},
    "HDF5": {"type": ["h5", "hdf5"], "reader": "custom_hdf5"}
}


selected_type = st.selectbox(
    "Select file type:",
    options=list(FILE_TYPES.keys()),
    index=0  
)

uploaded_file = st.file_uploader(
    f"Upload {selected_type} file",
    type=FILE_TYPES[selected_type]["type"]
)

if uploaded_file:
    try:
        if FILE_TYPES[selected_type]["reader"] not in ["custom_hdf5"]:
            df = FILE_TYPES[selected_type]["reader"](uploaded_file)
            success_placeholder = st.empty()
            success_placeholder.success("✅ File loaded successfully!")
            time.sleep(2)  
            success_placeholder.empty() 
            # To show the column names (not neccessary)
            # st.subheader("Column Names:")
            # st.write(list(df.columns))
            st.subheader("Sample from the data")
            st.write(df.head(5))
                
        elif selected_type == "HDF5":
            with h5py.File(uploaded_file, 'r') as f:
                st.write("HDF5 Groups/Datasets:", list(f.keys()))
                dataset_name = st.selectbox("Select dataset", list(f.keys()))
                data = f[dataset_name][:]
                st.subheader("Sample from the data")
                st.write("First 5 rows:", data[:5]) 
            
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")