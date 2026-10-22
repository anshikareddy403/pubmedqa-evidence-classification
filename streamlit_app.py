"""Wrong entry point if you have another project named streamlit_app.py — use pubmedqa_evidence_demo.py."""

import streamlit as st

st.set_page_config(page_title="Wrong app — use PubMedQA demo", page_icon="⚠️")
st.warning(
    "This file is a placeholder. For the **PubMedQA NLP project**, run the dedicated demo "
    "so it does not open your other Streamlit project on port 8501."
)
st.code(
    "streamlit run pubmedqa_evidence_demo.py --server.port 8510",
    language="bash",
)
st.markdown("Or double-click **`run_pubmedqa_demo.bat`** in this repo folder.")
st.markdown("Then open **http://localhost:8510** in your browser.")
