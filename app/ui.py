import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import os
# Configuration
#API_URL = "http://127.0.0.1:8000"  # Adjust if your FastAPI runs elsewhere
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="ML Submission Portal", layout="wide")

st.title("🛡️ Model & Data Quality Dashboard")
st.markdown("Upload your model artifacts and data to trigger automated quality checks.")

# --- Sidebar: Identity & Selection ---
with st.sidebar:
    st.header("Submission Management")
    
    # Fetch existing IDs from the API
    try:
        response = requests.get(f"{API_URL}/submissions")
        if response.status_code == 200:
            existing_ids = response.json()
        else:
            existing_ids = []
    except Exception:
        existing_ids = []
        st.error("Could not connect to backend to fetch IDs.")

    # UI for selecting or creating an ID
    mode = st.radio("Mode", ["Select Existing", "Create New"])
    
    if mode == "Select Existing" and existing_ids:
        submission_id = st.selectbox("Choose a Submission ID", options=existing_ids)
    else:
        submission_id = st.text_input("Enter New Submission ID", value="user_001")
        if mode == "Select Existing" and not existing_ids:
            st.warning("No existing submissions found. Please create one.")

    st.divider()
    st.info(f"Active ID: **{submission_id}**")

# --- Tabs for different steps ---
tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "📊 Data Profiling", "🔍 Model Scan"])

# --- TAB 1: UPLOAD ---
with tab1:
    st.header("Upload Artifacts")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")

        st.info("Please upload your model exported to **ONNX** format (`.onnx`). "
                "PyTorch, XGBoost, scikit-learn, TensorFlow, etc. can all be exported to ONNX.")
        checkpoint_file = st.file_uploader("Upload Model Checkpoint", type=["onnx"])

        if st.button("Submit Model"):
            if checkpoint_file and submission_id:
                files = {"checkpoint_file": (checkpoint_file.name, checkpoint_file.getvalue())}
                params = {"submission_id": submission_id}

                with st.spinner("Uploading model..."):
                    res = requests.post(f"{API_URL}/upload/model", params=params, files=files)
                    if res.status_code == 200:
                        st.success("✅ ONNX model uploaded!")
            else:
                st.warning("Please select a file.")

    with col2:
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload data.csv", type=["csv"])
        
        if data_file:
            # Load preview to let user select targets
            df_preview = pd.read_csv(data_file)
            st.write("### Preview & Configure")
            st.dataframe(df_preview.head(3))
            
            all_cols = df_preview.columns.tolist()
            selected_targets = st.multiselect(
                "Select Target Column(s)", 
                options=all_cols,
                help="Any columns NOT selected here will be treated as features."
            )
            
            if st.button("Confirm & Submit Data"):
                if not selected_targets:
                    st.error("Please select at least one target column.")
                
                # Reset file pointer to beginning for upload
                data_file.seek(0)
                
                files = {"file": (data_file.name, data_file.getvalue())}
                # Pass targets as multiple query parameters
                params = [
                    ("submission_id", submission_id)
                ] + [("targets", t) for t in selected_targets]
                
                res = requests.post(f"{API_URL}/upload/data", params=params, files=files)
                
                if res.status_code == 200:
                    st.success(f"Data uploaded! Features: {len(all_cols)-len(selected_targets)}, Targets: {len(selected_targets)}")
                else:
                    st.error(f"Upload failed: {res.json().get('detail')}")

# --- TAB 2: DATA PROFILING ---
with tab2:
    st.header("Data Profiling Report")
    if st.button("Generate Profiling Report"):
        with st.spinner("Running ydata-profiling..."):
            res = requests.get(f"{API_URL}/check_data", params={"submission_id": submission_id})
            if res.status_code == 200:
                st.components.v1.html(res.text, height=800, scrolling=True)
            else:
                st.warning(res.json().get("detail", "Error fetching report"))

# --- TAB 3: MODEL SCAN ---
with tab3:
    st.header("Model Scan & Simulation")

    scan_mode = st.radio(
        "Select Scan Type",
        ["Standard Model Scan (Giskard)", "IEEE 39-Bus Physical Simulation"]
    )

    # -------------------------
    # STANDARD MODEL SCAN
    # -------------------------
    if scan_mode == "Standard Model Scan (Giskard)":
        st.markdown("Scans for robustness, bias, and data leakage (EU AI Act Compliance).")

        if st.button("Run Comprehensive Scan"):
            with st.spinner("Scanning model (this may take a minute)..."):
                try:
                    res = requests.get(
                        f"{API_URL}/check_model",
                        params={"submission_id": submission_id}
                    )
                    if res.status_code == 200:
                        st.components.v1.html(res.text, height=1000, scrolling=True)
                    else:
                        st.error(res.json().get("detail", "Error during scan"))
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    # -------------------------
    # IEEE BUS 39 SIMULATION
    # ------------------------- 
    else:
        st.info("⚡ Configure IEEE 39-Bus physical grid simulation.")
        params = None
        with st.form("ieee_params"):
            i_max_ka = st.number_input("Max Line Current (i_max_ka)", value=1.2, step=0.1)
            vmin = st.number_input("Min Voltage (v_min_pu)", value=0.95, step=0.01)
            vmax = st.number_input("Max Voltage (v_max_pu)", value=1.05, step=0.01)
            submited = st.form_submit_button()
            
            if submited: 
                params = {
                    'submission_id': submission_id,
                    'i_max_ka': i_max_ka,
                    'vmin': vmin,
                    'vmax': vmax
                }
                res = requests.post(f"{API_URL}/upload/ieee_bus39_config", params=params)
                if res.status_code == 200:
                    st.write("Submitted")
                else:
                     st.warning(res.json().get("detail", "Error writing ieeebus39 config"))
                
                
    
        
