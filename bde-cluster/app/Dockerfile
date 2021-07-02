FROM bde2020/spark-python-template:3.1.1-hadoop3.2

RUN apk add --no-cache py3-numpy

ENV SPARK_MASTER spark://pyspark-master:7067
ENV SPARK_APPLICATION_PYTHON_LOCATION app/app.py
ENV SPARK_SUBMIT_ARGS "--total-executor-cores 80 --executor-memory 16g --executor-cores 8"
ENV HDFS_ROOT hdfs://namenode:9000
ENV HDFS_DATASET_PATH /data/data/

ADD start.sh /

RUN chmod +x /start.sh

CMD ["/bin/bash", "/start.sh"]