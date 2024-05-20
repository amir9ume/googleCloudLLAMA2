# Use an official Python runtime as a parent image
#FROM python:3.9-slim-buster
FROM --platform=linux/amd64 python:3.8-slim-buster as build

# Set the working directory in the container
WORKDIR /app

#COPY ~/.cache/huggingface/hub/models--microsoft--phi-2 /app

# Install gcc and other necessary build tools
RUN apt-get update && apt-get install -y gcc python3-dev

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the model files to the container
COPY ./models /app/models


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run Gunicorn when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "transformer:app"]





# Run transformer.py when the container launches
#CMD ["python", "transformer.py"]








