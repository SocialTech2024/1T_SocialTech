FROM python:3.13.1
RUN apt-get update
COPY /app /app
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "-m", "uvicorn", "app.main:app", "--reload" ]