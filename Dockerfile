From python:3.11
Workdir /app
Copy requirements.txt .
RUN pip install --upgrade pip
Run pip install -r requirements.txt
Copy . .
Expose 5000
CMD ["python","app.py"]
