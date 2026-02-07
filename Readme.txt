env setup ->

cd /d "ABS_PATH"
.venv\Scripts\activate
where python
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
python -m pip install streamlit joblib numpy pandas scikit-learn xgboost


Run -> 
streamlit run app.py