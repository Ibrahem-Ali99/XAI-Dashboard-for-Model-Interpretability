import streamlit as st
import pandas as pd
import requests
import h5py

FLASK_URL = "http://localhost:5000"

# File type configurations
FILE_TYPES = {
    "CSV": {"type": ["csv"], "reader": pd.read_csv},
    "Excel": {"type": ["xlsx", "xls"], "reader": pd.read_excel},
    "JSON": {"type": ["json"], "reader": pd.read_json},
    "Pickle (Python)": {"type": ["pkl"], "reader": pd.read_pickle},
    "HDF5": {"type": ["h5", "hdf5"], "reader": "custom_hdf5"}
}

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.target = None
    st.session_state.task = None
    st.session_state.model = None
    st.session_state.training_complete = False

# File upload section
st.title("Machine Learning Model Trainer")
selected_type = st.selectbox("Select file type:", options=list(FILE_TYPES.keys()))
uploaded_file = st.file_uploader(f"Upload {selected_type} file", type=FILE_TYPES[selected_type]["type"])

if uploaded_file:
    try:
        if FILE_TYPES[selected_type]["reader"] != "custom_hdf5":
            df = FILE_TYPES[selected_type]["reader"](uploaded_file)
            st.session_state.df = df
            st.success("File loaded successfully!", icon="✅")
            
            # Add warning about categorical features
            object_cols = df.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                st.warning(f"Found {len(object_cols)} categorical columns that will be automatically encoded", icon="⚠️")
                
            st.subheader("Sample Data")
            st.write(df.head(5))
            
            # Target variable selection
            st.session_state.target = st.selectbox(
                "Select Target Variable",
                options=df.columns,
                index=len(df.columns)-1
            )
            
        elif selected_type == "HDF5":
            with h5py.File(uploaded_file, 'r') as f:
                dataset_name = st.selectbox("Select dataset", list(f.keys()))
                df = pd.DataFrame(f[dataset_name][:])
                st.session_state.df = df
                st.subheader("Sample Data")
                st.write("First 5 rows:", df.head(5))
                
    except Exception as e:
        st.error(f"Error loading file: {str(e)}", icon="❌")

# Model selection section
if st.session_state.df is not None and st.session_state.target:
    st.subheader("Model Configuration")
    
    # Task selection
    task = st.radio(
        "Select Task Type:",
        ["Classification", "Regression"],
        horizontal=True
    )
    
    # Model options based on task
    classification_models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    regression_models = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
    
    selected_model = st.selectbox(
        "Choose Model",
        classification_models if task == "Classification" else regression_models
    )
    
    # Proceed button
    if st.button("Train Model", use_container_width=True):
        st.session_state.task = task
        st.session_state.model = selected_model
        
        # Prepare data for training
        X = st.session_state.df.drop(columns=[st.session_state.target])
        y = st.session_state.df[st.session_state.target]
        
        # Send to Flask backend
        try:
            response = requests.post(
                f"{FLASK_URL}/train",
                json={
                    "X": X.to_dict(orient='records'),
                    "y": y.tolist(),
                    "task": task,
                    "model": selected_model
                }
            )
            
            # ... (after training button click)
            if response.status_code == 200:
                result = response.json()
                st.session_state.metrics = result['metrics']
                st.session_state.model_id = result['model_id']
                st.success("Training Complete!", icon="✅")
                
                # Show metrics based on task type
                st.subheader("Training Results")
                metrics = result['metrics']
                
                if task == "Classification":
                    st.write(f"Accuracy: {metrics.get('accuracy', 0):.2f}")
                    st.write(f"F1 Score: {metrics.get('f1_score', 0):.2f}")
                    st.write(f"Precision: {metrics.get('precision', 0):.2f}")
                    st.write(f"Recall: {metrics.get('recall', 0):.2f}")
                else:
                    st.write(f"MSE: {metrics.get('mse', 0):.2f}")
                    st.write(f"RMSE: {metrics.get('rmse', 0):.2f}")
                    st.write(f"R² Score: {metrics.get('r2', 0):.2f}")
                
            else:
                st.error(f"Training failed: {response.text}", icon="❌")
                
        except Exception as e:
            st.error(f"Connection error: {str(e)}", icon="❌")