FROM python:3.10-slim

WORKDIR /app

# Copying only the requirements file first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the application code
COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app"]
