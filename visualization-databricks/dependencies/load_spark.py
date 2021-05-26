from pyspark.sql import SparkSession


def load_spark():
    spark = SparkSession.builder.appName('visualization').getOrCreate()

    return spark


def get_db_utils(spark):
    if spark.conf.get("spark.databricks.service.client.enabled") == "true":
        from pyspark.dbutils import DBUtils
        return DBUtils(spark)
    else:
        import IPython
        return IPython.get_ipython().user_ns["dbutils"]
