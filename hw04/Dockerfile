FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

COPY Pipfile Pipfile.lock ./

RUN pip install pandas && pip install scikit-learn==1.5.0 && pip install pyarrow

ENTRYPOINT ["python", "starter.py"]
CMD []
