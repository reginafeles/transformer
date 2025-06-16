FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install flask torch

EXPOSE 7860

CMD ["python", "app.py"]
