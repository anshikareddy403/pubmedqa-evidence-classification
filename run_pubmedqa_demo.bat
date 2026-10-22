@echo off
REM PubMedQA evidence demo — port 8510 (not default 8501)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
streamlit run pubmedqa_evidence_demo.py --server.port 8510
