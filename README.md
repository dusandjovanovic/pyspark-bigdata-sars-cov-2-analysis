## Struktura analiza, odnosno `job`-ova

Zbog lakšeg izvršavanja i testiranja korak 'Transformacije' izolovan je od 'Ekstrakcije' i 'učitavanja' - ulazni podaci se prihvataju i pakuju u jedinstveni DataFrame. Zatim, kod koji obuhvata transformacije u `main()` funkciji, bavi se izvlačenjem podataka, daljim prosledjivanje funkciji transformacije, kao i čuvanjem rezultata.

Generalizovano, funkcije transformacije bi trebalo dizajnirati kao _idempotent_ funkcije. Drugim rečima, višestruko primenjianje funkcija transformacije ne bi trebalo da rezultuje promenama u izlazu, sve dok nema promena ulaznih podataka. Zbog ovakvog pristupa moguće je izvršavati analize sa ponavlanjima ukoliko je to potrebno (na primer, koristeći `cron` za poziv `spark-submit` komande, po pre-definisanom rasporedu poziva).

## Prosledjivanje konfiguracionih parametara analizama

Kako se ne bi slali argumenti sa komandne linije, efiaksnije rešenje je koristiti konfiguracione fajlove po potrebi - na primer, koristeći `--files configs/jon_name_config.json` flag sa `spark-submit` komandom - flag koji će referencirati konfiguracionu datoteku, datoteka koja se može koristiti u analizama u vidu rečnika, iliti `dictionary` ulaza kao `json.loads(config_file_contents)`.

```python
import json

config = json.loads("""{"field": "value"}""")
```

Datoteka se učitava i parsuje funkcijom `start_spark()` iz pomoćne datoteke `dependencies/spark.py` koja pored parsovanja konfiguracionih fajlova učitava Spark drajver program koji se pokreće na klasteru i alocira loger.

## Pakovanje dependency-a analiza

Deljene funkcije na koje se oslanjaju analize nalaze se u paketu `dependencies` i referenciraju module Spark-a koji su neophodni, na primer:

```python
from dependencies.spark import start_spark
```

Ovaj paket, zajedno sa svim ostalim dependency-ma, mora biti kopiran na svaki Spark čvor. Postoji više načina za postizanje ovoga, izabrano je pakovanje svih zavisnosti u `zip` arhivu zajedno sa poslom koji treba izvršiti, zatim se koristi `--py-files` naredba prilikom pokretanja analize. Pomoćna shell skripta `build_dependencies.sh` koristi se za pakovanje arhive. Ova skripta uzima u obzir graf zavisnosti okruženja i sve navedene zavisnosti u `Pipfile` datoteci.

## Pokretanje posla lokalno/na klasteru

Pokretanje Spark klastera lokalno:

```bash
cd $SPARK_HOME && ./spark-shell --master local
```

Ukoliko `$SPARK_HOME` promenljiva okruženja ukazuje na instalaciju Spark-a, analiza (posao) pokreće se:

```bash
$SPARK_HOME/bin/spark-submit \
--master local \
--packages 'com.somesparkjar.dependency:1.0.0' \
--py-files packages.zip \
--files configs/etl_config.json \
jobs/job_name.py
```

- `--master local[*]` - adresa Spark klastera. Ovo može biti lokalni klaster ili klaster u cloud-u koji se zadaje adresom `spark://adresa_klastera:7077`;
- `--packages 'com.somesparkjar.dependency:1.0.0,...'` - Opcionalno Maven dependency lista koja je potrebna za izvršavanje;
- `--files configs/etl_config.json` - opcionalno putanja do konfiguracione datoteke;
- `--py-files packages.zip` - prethodno pomenuta arhiva sa dependency-ma;
- `jobs/job_name.py` - Python modul sa kodom analize/posla.