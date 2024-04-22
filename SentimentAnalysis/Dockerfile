FROM python:3.11.3
COPY . .

RUN pip install nltk
RUN python -m nltk.downloader stopwords wordnet punkt
RUN pip3 install -r requirements.txt
CMD ["deployment.py"]
ENTRYPOINT [ "python" ]

