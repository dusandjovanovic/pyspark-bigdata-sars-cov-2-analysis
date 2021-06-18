import __main__

from os import listdir, path
import json
from pyspark import SparkFiles, SQLContext
from pyspark.sql import SparkSession
from dependencies import logging


def start_spark(app_name='sars_cov2_analysis', master='local[*]', jar_packages=[], files=[], spark_config={}):
    flag_repl = not (hasattr(__main__, '__file__'))

    # ručno podesiti debug ukoliko je potrebno
    # eventualno .env sa 'DEBUG' in environ.keys()
    flag_debug = False

    if not (flag_repl or flag_debug):
        spark_builder = (
            SparkSession
                .builder
                .appName(app_name))
    else:
        spark_builder = (
            SparkSession
                .builder
                .master(master)
                .appName(app_name))

        spark_jars_packages = ','.join(list(jar_packages))
        spark_builder.config('spark.jars.packages', spark_jars_packages)

        spark_files = ','.join(list(files))
        spark_builder.config('spark.files', spark_files)

        for key, val in spark_config.items():
            spark_builder.config(key, val)

    spark_builder.config("spark.executor.memory", "2g")

    spark_sess = spark_builder.getOrCreate()
    spark_logger = logging.Log4j(spark_sess)

    spark_files_dir = SparkFiles.getRootDirectory()
    config_files = [filename
                    for filename in listdir(spark_files_dir)
                    if filename.endswith('config.json')]

    if config_files:
        path_to_config_file = path.join(spark_files_dir, config_files[0])
        with open(path_to_config_file, 'r') as config_file:
            config_dict = json.load(config_file)
        spark_logger.warn('Config: ' + config_files[0])
    else:
        spark_logger.warn('Warning: No config found.')
        config_dict = None

    sql_context = SQLContext(spark_sess.sparkContext)

    return spark_sess, sql_context, spark_logger, config_dict
