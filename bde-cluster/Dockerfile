FROM bde2020/spark-submit:3.1.1-hadoop3.2

# Add build dependencies for c-libraries (important for building numpy and other sci-libs)
RUN apk --no-cache add --virtual build-deps musl-dev linux-headers g++ gcc python3-dev

# Copy the requirements.txt first, for separate dependency resolving and downloading
COPY app/requirements.txt /app/
RUN cd /app \ && pip3 install -r requirements.txt

# Copy the source code
COPY app /app
ADD start_app.sh /
RUN chmod +x /start_app.sh

ENV SPARK_MASTER spark://spark-master:7077
ENV SPARK_APPLICATION_PYTHON_LOCATION app/app.py
ENV HDFS_ROOT hdfs://namenode:9000
ENV HDFS_DATASET_PATH /data/data/

CMD ["/bin/bash", "/start_app.sh"]