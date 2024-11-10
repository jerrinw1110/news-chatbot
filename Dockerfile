# Dockerfile, Image, Container
FROM python:3.12

ADD main.py summaries.py Bert.py /

RUN pip install openai

RUN pip install python-dotenv

RUN pip install bert-score

CMD ["python", "main.py"]