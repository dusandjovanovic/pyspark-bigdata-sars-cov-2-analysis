## Izvršavanje na klasteru računara `Big-data Europe`

#### Pokretanje infrastrukture i kontejnera

Infrastruktura je opisana u odvojenoj konfiguracionoj datoteci i sadrži više kontejnera poput `namenode`, `datanote`, `historyserver` itd. Potrebno je podići infrastrukturu, a zatim postaviti dataset na HDFs. Nakon ovih priprema, može se pokrenuti kontejner analize.

* `$ docker network create bde`
* `$ cd infrastructure && ./infra_start.sh && ./infra_upload_to_hdfs.sh`
* `$ cd .. && ./analysis_start.sh`

#### Infrastruktura

Infrastruktura sadrži potrebne *image*, ovo podrazumeva neophodne *Hadoop* deamone, kao i master/worker čvorove.

```yaml
services:
  pyspark-master:
    image: bde2020/spark-master:3.1.1-hadoop3.2
    container_name: pyspark-master
    ...
  pyspark-worker-1:
    image: bde2020/spark-worker:3.1.1-hadoop3.2
    container_name: pyspark-worker-1
    ...
  namenode:
    image: bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8
    container_name: namenode
    ...
  datanode:
    image: bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8
    container_name: datanode
    ...
  resourcemanager:
    image: bde2020/hadoop-resourcemanager:2.0.0-hadoop3.2.1-java8
    container_name: resourcemanager
    ...
  nodemanager1:
    image: bde2020/hadoop-nodemanager:2.0.0-hadoop3.2.1-java8
    container_name: nodemanager
    ...
  historyserver:
    image: bde2020/hadoop-historyserver:2.0.0-hadoop3.2.1-java8
    container_name: historyserver
    ...

volumes:
  hadoop_namenode:
  hadoop_datanode:
  hadoop_historyserver:

networks:
  default:
    external:
      name: bde
```

#### "Submit" kontejner

Osnovni imidž koji se proširava je `bde2020/spark-submit`.

Odavde se poziva *spark-submit* komanda i python izvršna datoteka šalje ka klasteru. Na početku je potrebno dodati zavisnosti i alate za kompajliranje .c slojeva biblioteka (poput `gcc` kompajlera). U ovom slučaju, analize se zbog algoritama mašinskog učenja oslanjaju na `numpy` biblioteku.

Zatim se na osnovu niza zavisnosti pribavljaju potrebni izvori i razrešavaju zavisnosti. Na kraju, poziva se *submit* naredba izvršnog programa `/app/app.py`. Ukoliko je neophodno mogu se definisati promenljive okruženja poput `HDFS_ROOT` i `HDFS_DATASET_PATH`.

```Dockerfile
FROM bde2020/spark-submit:3.1.1-hadoop3.2

# Add build dependencies for c-libraries (important for building numpy and other sci-libs)
RUN apk --no-cache add --virtual build-deps musl-dev linux-headers g++ gcc python3-dev

# Copy the requirements.txt first, for separate dependency resolving and downloading
COPY app/requirements.txt /app/
RUN cd /app \ && pip3 install -r requirements.txt

# Copy the source code
COPY app /app
ADD app/start.sh /
RUN chmod +x /start_app.sh

ENV SPARK_MASTER spark://spark-master:7077
ENV SPARK_APPLICATION_PYTHON_LOCATION app/app.py
ENV SPARK_SUBMIT_ARGS "--total-executor-cores 80 --executor-memory 16g --executor-cores 8"
ENV HDFS_ROOT hdfs://namenode:9000
ENV HDFS_DATASET_PATH /data/data/

CMD ["/bin/bash", "/start_app.sh"]
```

Na osnovu Dockerfile-a se izgradjuje kontejner. Potrebno je da svi kontejneri definisani u `infrastructure`/`submit` imaju pristup isto deljenoj mreži, u ovom slučaju nazvanoj `bde`.

```yml
version: "3"

services:
  submit:
    build: ./
    image: spark-submit:latest
    container_name: analysis
    environment:
      SPARK_MASTER_NAME: spark-master
      SPARK_MASTER_PORT: 7077
      SPARK_APPLICATION_ARGS: ""
      ENABLE_INIT_DAEMON: "false"

networks:
  default:
    external:
      name: bde
```
