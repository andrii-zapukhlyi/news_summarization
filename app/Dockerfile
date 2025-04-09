FROM python:3.13-alpine
WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]