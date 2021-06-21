## Izvršavanje na klasteru računara `Big-data Europe`

#### Docker

Osnovni imidž koji se proširava je `bde2020/spark-submit`.

Odavde se poziva *spark-submit* komanda i python izvršna datoteka šalje ka klasteru. Na početku je potrebno dodati zavisnosti i alate za kompajliranje .c slojeva biblioteka (poput `gcc` kompajlera). U ovom slučaju, analize se zbog algoritama mašinskog učenja oslanjaju na `numpy` biblioteku.

Zatim se na osnovu niza zavisnosti pribavljaju potrebni izvori i razrešavaju zavisnosti. Na kraju, poziva se *submit* naredba izvršnog programa `/app/app.py`. Ukoliko je neophodno mogu se definisati promenljive okruženja poput `HDFS_ROOT` i `HDFS_DATASET_PATH`.

```bash
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
```

Na osnovu Dockerfile-a se izgradjuje kontejner. Za lokalna testiranja je važno postaviti deljenu mrežu, u ovom slučaju nazvanu `bde`. Deljena mreža treba da se poklapa sa mrežom u `/infrastructure/compose.yml`.

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


#### Pokretanje Docker kontejnera

Infrastruktura je opisana u odvojenoj konfiguracionoj datoteci i sadrži više kontejnera poput `namenode`, `datanote`, `historyserver` itd. Potrebno je podići infrastrukturu, a zatim postaviti dataset na HDFs. Nakon ovih priprema, može se pokrenuti kontejner analize.

* `$ docker network create bde`
* `$ cd infrastructure && ./start_spark.sh && ./upload_dataset_hdfs.sh`
* `$ cd .. && ./build_analysis.sh && ./start_analysis.sh`
