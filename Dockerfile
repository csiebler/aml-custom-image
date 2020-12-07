FROM mcr.microsoft.com/azureml/base:intelmpi2018.3-ubuntu16.04

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt