import sys

import numpy as np
import pyspark.sql.functions as func
from pyspark.sql.functions import udf
from functools import reduce
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='radiography_analysis',
        files=['configs/radiography_analysis_config.json'])

    log.warn('Running radiography analysis...')

    # extracting and transforming the dataset
    [data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia] = extract_data(spark)
    data_initial = transform_data(data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia, sql_context)

    # # percentage of samples (different categories)
    # data_transformed = transform_percentage_of_samples(data_initial)
    # load_data(data_transformed, "percentage_of_samples")
    #
    # # take one sample of each group
    # data_transformed = transform_take_samples(data_initial)
    # load_data(data_transformed, "take_samples")
    #
    # # colour distribution
    # data_transformed = transform_colour_distribution(data_initial)
    # load_data(data_transformed, "colour_distribution")

    # ML classification
    data_transformed = transform_ml_classification(data_initial)
    load_data(data_transformed, "ml_classification")

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

    dataframe_repartitioned = dataframe_merged.repartition(200)

    return dataframe_merged


def transform_percentage_of_samples(dataframe):
    df_percentages = dataframe.groupby('label') \
        .agg((func.count('image')).alias('count'), (func.count('image') / dataframe.count()).alias('percentage')) \
        .orderBy(func.col("label").asc())

    return df_percentages


def transform_take_samples(dataframe):
    df_samples = dataframe.dropDuplicates(['label'])
    df_samples = df_samples.withColumn("encoded", func.base64(func.col("image.data")))

    return df_samples


def transform_colour_distribution(dataframe):
    udf_function_min = udf(min_value, FloatType())
    udf_function_max = udf(max_value, FloatType())
    udf_function_mean = udf(mean_value, FloatType())
    udf_function_standard_deviation = udf(standard_deviation_value, FloatType())

    sample_size = 1000

    df_normal = dataframe \
        .filter(dataframe.label == DESCRIPTOR_NORMAL) \
        .limit(sample_size)

    df_covid19 = dataframe \
        .filter(dataframe.label == DESCRIPTOR_COVID19) \
        .limit(sample_size)

    df_lung_opacity = dataframe \
        .filter(dataframe.label == DESCRIPTOR_LUNG_OPACITY) \
        .limit(sample_size)

    df_viral_pneumonia = dataframe \
        .filter(dataframe.label == DESCRIPTOR_VIRAL_PNEUMONIA) \
        .limit(sample_size)

    df_merged = reduce(lambda first, second: first.union(second),
                       [df_normal, df_covid19, df_lung_opacity, df_viral_pneumonia])

    df_colour_distribution = df_merged \
        .withColumn("min", udf_function_min("image.data")) \
        .withColumn("max", udf_function_max("image.data")) \
        .withColumn("mean", udf_function_mean("image.data")) \
        .withColumn("standard_deviation", udf_function_standard_deviation("image.data"))

    return df_colour_distribution


def transform_ml_classification(dataframe):
    # model: InceptionV3
    # extracting feature from images
    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")

    # used as a multi class classifier
    classifier = LogisticRegression(maxIter=5, regParam=0.03, elasticNetParam=0.5, labelCol="label")

    # split the dataset
    train, test = dataframe.randomSplit([0.8, 0.2], 42)

    # define a pipeline model
    model = Pipeline(stages=[featurizer, classifier])
    spark_model = model.fit(train)

    evaluator = MulticlassClassificationEvaluator()
    transform_test = spark_model.transform(test)

    print('F1-Score ', evaluator.evaluate(transform_test, {evaluator.metricName: 'f1'}))
    print('Precision ', evaluator.evaluate(transform_test, {evaluator.metricName: 'weightedPrecision'}))
    print('Recall ', evaluator.evaluate(transform_test, {evaluator.metricName: 'weightedRecall'}))
    print('Accuracy ', evaluator.evaluate(transform_test, {evaluator.metricName: 'accuracy'}))

    return dataframe


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .json("./outputs/radiography_analysis/" + name, mode='overwrite'))
    return None


def mean_value(arr):
    return float(np.mean(arr))


def min_value(arr):
    return float(np.min(arr))


def max_value(arr):
    return float(np.max(arr))


def standard_deviation_value(arr):
    return float(np.std(arr))


DESCRIPTOR_NORMAL = 0
DESCRIPTOR_COVID19 = 1
DESCRIPTOR_LUNG_OPACITY = 2
DESCRIPTOR_VIRAL_PNEUMONIA = 3

if __name__ == '__main__':
    main()
