FROM python:3.9-slim

# setting the working directory
WORKDIR /app

# copying project files
COPY . /app

# installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# exposing the Flask app port
EXPOSE 5000

# running the Flask app
CMD ["python", "app.py"]
