import sys

import pyspark.sql.functions as func
from functools import reduce

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='radiography_analysis',
        files=['configs/radiography_analysis_config.json'])

    log.warn('Running radiography analysis...')

    # extracting and transforming the dataset
    [data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia] = extract_data(spark)
    data_initial = transform_data(data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia, sql_context)

    # percentage of samples (different categories)
    data_transformed = transform_percentage_of_samples(data_initial)
    load_data(data_transformed, "percentage_of_samples")

    log.warn('Terminating radiography analysis...')

    spark.stop()
    return None


def extract_data(spark):
    normal_image_dir = sys.argv[1] + "/Normal/"
    covid19_image_dir = sys.argv[1] + "/COVID/"
    lung_opacity_image_dir = sys.argv[1] + "/Lung_Opacity/"
    viral_pneumonia_image_dir = sys.argv[1] + "/Viral_Pneumonia/"

    dataframe_normal = spark.read.format("image").option("dropInvalid", True) \
        .load(normal_image_dir).withColumn("label", func.lit(DESCRIPTOR_NORMAL))

    dataframe_covid19 = spark.read.format("image").option("dropInvalid", True) \
        .load(covid19_image_dir).withColumn("label", func.lit(DESCRIPTOR_COVID19))

    dataframe_lung_opacity = spark.read.format("image").option("dropInvalid", True) \
        .load(lung_opacity_image_dir).withColumn("label", func.lit(DESCRIPTOR_LUNG_OPACITY))

    dataframe_viral_pneumonia = spark.read.format("image").option("dropInvalid", True) \
        .load(viral_pneumonia_image_dir).withColumn("label", func.lit(DESCRIPTOR_VIRAL_PNEUMONIA))

    return [dataframe_normal, dataframe_covid19, dataframe_lung_opacity, dataframe_viral_pneumonia]


def transform_data(data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia, sql_context):
    dataframe_merged = reduce(lambda first, second: first.union(second),
                              [data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia])

    # dataframe_repartitioned = dataframe_merged.repartition(200)

    return dataframe_merged


def transform_percentage_of_samples(dataframe):
    df_percentages = dataframe.groupby('label') \
        .agg((func.count('image')).alias('count'), (func.count('image') / dataframe.count()).alias('percentage')) \
        .orderBy(func.col("label").asc())

    return df_percentages


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .json("./outputs/radiography_analysis/" + name, mode='overwrite'))
    return None


DESCRIPTOR_NORMAL = 0
DESCRIPTOR_COVID19 = 1
DESCRIPTOR_LUNG_OPACITY = 2
DESCRIPTOR_VIRAL_PNEUMONIA = 3

if __name__ == '__main__':
    main()
