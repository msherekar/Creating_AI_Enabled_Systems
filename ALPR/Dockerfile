FROM python:3.11.7
RUN apt update
RUN apt install -y ffmpeg
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev
RUN pip install pytesseract
RUN pip install matplotlib
RUN pip install ffmpeg-python opencv-python-headless
 
 
COPY . .
CMD ["deployment.py"]
ENTRYPOINT [ "python" ]