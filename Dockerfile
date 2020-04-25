FROM python:3
COPY . /app
WORKDIR /app
RUN pip install pandas streamlit  matplotlib lightgbm altair sklearn
CMD streamlit run ./demo_main.py 
EXPOSE 8501 
