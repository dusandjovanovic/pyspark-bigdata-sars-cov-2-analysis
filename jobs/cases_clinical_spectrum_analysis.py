import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark.sql.functions as func
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyspark.sql.types import IntegerType
import re


def load_raw():
    spark = SparkSession.builder \
        .master('local') \
        .appName('myAppName') \
        .config('spark.executor.memory', '12gb') \
        .config("spark.cores.max", "10") \
        .getOrCreate()

    sc = spark.sparkContext

    sqlContext = SQLContext(sc)

    df = pd.read_excel('../data/covid-19-clinical-spectrum/dataset.xlsx')

    df['Respiratory Syncytial Virus'] = df['Respiratory Syncytial Virus'].astype(str)
    df['Influenza A'] = df['Influenza A'].astype(str)
    df['Influenza B'] = df['Influenza B'].astype(str)
    df['Parainfluenza 1'] = df['Parainfluenza 1'].astype(str)
    df['CoronavirusNL63'] = df['CoronavirusNL63'].astype(str)
    df['Rhinovirus/Enterovirus'] = df['Rhinovirus/Enterovirus'].astype(str)
    df['Coronavirus HKU1'] = df['Coronavirus HKU1'].astype(str)

    for column in df.columns:
        df[column] = df[column].astype(str)

    df = sqlContext.createDataFrame(df)

    return df, sqlContext


df, sqlContext = load_raw()

df = df.fillna(0)
df = df.replace("nan", "0")

df_hemoglobin = df.select("Hemoglobin").toPandas()
df_hemoglobin['Hemoglobin'] = pd.to_numeric(df_hemoglobin['Hemoglobin'])
df_hemoglobin['Hemoglobin'].hist()

df.select("SARS-Cov-2 exam result").show()

df_ = df.select(func.col("SARS-Cov-2 exam result").alias("result"), func.col('Patient age quantile').alias('age'))
df_.show()

df_.select("result", "age").write.mode('overwrite').option("header", "true").save("result_age.parquet",
                                                                                  format="parquet")

df_ = sqlContext.sql("SELECT * FROM parquet.`./result_age.parquet`")

df_pandas_age = df_.groupBy("result").agg(func.max("age"), func.avg("age")).toPandas()
df_pandas_age.plot()

columns = df.schema.names
for column in columns:
    df = df.withColumn(column, df[column].cast(IntegerType()))

df_numeric_pandas = df.toPandas()
df_class_1 = df_numeric_pandas[df_numeric_pandas['SARS-Cov-2 exam result'] != 'negative']
df_class_0 = df_numeric_pandas[df_numeric_pandas['SARS-Cov-2 exam result'] == 'negative']
df_class_0 = df_class_0[:len(df_class_1)]

df_numeric_concat = pd.concat([df_class_0, df_class_1], axis=0)

y = df_numeric_concat['SARS-Cov-2 exam result']

y_l = [0 if r == 'negative' else 1 for r in y]

columns_to_drop = ['SARS-Cov-2 exam result', 'Patient ID']
X = df_numeric_concat.drop('SARS-Cov-2 exam result', axis=1)

columns = X.columns
X.columns = [str(re.sub(r"[^a-zA-Z0-9]+", ' ', column)) for column in columns]
columns = X.columns
X.columns = [column.replace("!@#$%^&*()[]{};:,./<>?\|`~-=_+", " ") for column in columns]

columns = X.columns

for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors='ignore')

X = pd.get_dummies(X)

for column in X.columns:
    if '<' in column:
        X = X.drop([column], axis=1)
