FROM python:3.8.5

EXPOSE 8501

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir


COPY MyApp.py ./MyApp.py
COPY data /data

CMD streamlit run MyApp.py