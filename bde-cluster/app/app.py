import os
import numpy as np

from functools import reduce
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.linalg import VectorUDT, DenseVector


def main():
    spark = SparkSession.builder.appName('radiography_analysis').getOrCreate()

    # extracting and transforming the dataset
    [data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia] = extract_data(spark)
    data_initial = transform_data(data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia)

    # percentage of samples (different categories)
    data_transformed = transform_percentage_of_samples(data_initial)
    load_data(data_transformed, "percentage_of_samples")

    # take one sample of each group
    data_transformed = transform_take_samples(data_initial)
    load_data(data_transformed, "take_samples")

    # colour distribution
    data_transformed = transform_colour_distribution(data_initial)
    load_data(data_transformed, "colour_distribution")

    # ML classification (distributed)
    data_transformed = transform_ml_classification(data_initial, spark)
    load_data(data_transformed, "ml_classification")

    spark.stop()

    return None


def extract_data(spark):
    root = os.getenv('HDFS_ROOT')
    dataset_location = os.getenv('HDFS_DATASET_PATH')

    normal_image_dir = root + dataset_location + "/Normal/"
    covid19_image_dir = root + dataset_location + "/COVID/"
    lung_opacity_image_dir = root + dataset_location + "/Lung_Opacity/"
    viral_pneumonia_image_dir = root + dataset_location + "/Viral_Pneumonia/"

    dataframe_normal = spark.read.format("image").option("dropInvalid", True) \
        .load(normal_image_dir).withColumn("label", func.lit(DESCRIPTOR_NORMAL))

    dataframe_covid19 = spark.read.format("image").option("dropInvalid", True) \
        .load(covid19_image_dir).withColumn("label", func.lit(DESCRIPTOR_COVID19))

    dataframe_lung_opacity = spark.read.format("image").option("dropInvalid", True) \
        .load(lung_opacity_image_dir).withColumn("label", func.lit(DESCRIPTOR_LUNG_OPACITY))

    dataframe_viral_pneumonia = spark.read.format("image").option("dropInvalid", True) \
        .load(viral_pneumonia_image_dir).withColumn("label", func.lit(DESCRIPTOR_VIRAL_PNEUMONIA))

    return [dataframe_normal, dataframe_covid19, dataframe_lung_opacity, dataframe_viral_pneumonia]


def transform_data(data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia):
    dataframe_merged = reduce(
        lambda first, second: first.union(second),
        [data_normal, data_covid19, data_lung_opacity, data_viral_pneumonia]
    )

    dataframe_merged = dataframe_merged.where(
        (func.col("image.height") == 299) & (func.col("image.width") == 299)
    )

    dataframe_repartitioned = dataframe_merged.repartition(200)

    return dataframe_repartitioned


def transform_percentage_of_samples(dataframe):
    df_percentages = dataframe.groupby('label') \
        .agg((func.count('image')).alias('count'), (func.count('image') / dataframe.count()).alias('percentage')) \
        .orderBy(func.col("label").asc())

    return df_percentages


def transform_take_samples(dataframe):
    udf_function_get_hdfs_origin = udf(hdfs_origin, StringType())
    udf_function_classify = udf(classify, StringType())

    df_samples = dataframe.dropDuplicates(['label']) \
        .withColumn("origin", udf_function_get_hdfs_origin("image")) \
        .withColumn("class_name", udf_function_classify("label")) \
        .drop("image", "label")

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

    df_merged = reduce(
        lambda first, second: first.union(second),
        [df_normal, df_covid19, df_lung_opacity, df_viral_pneumonia]
    )

    df_colour_distribution = df_merged \
        .withColumn("min", udf_function_min("image.data")) \
        .withColumn("max", udf_function_max("image.data")) \
        .withColumn("mean", udf_function_mean("image.data")) \
        .withColumn("standard_deviation", udf_function_standard_deviation("image.data"))

    return df_colour_distribution


def transform_ml_classification(dataframe, spark):
    udf_extract_features = udf(features_vector, VectorUDT())
    udf_function_min = udf(min_value, FloatType())
    udf_function_max = udf(max_value, FloatType())
    udf_function_mean = udf(mean_value, FloatType())
    udf_function_standard_deviation = udf(standard_deviation_value, FloatType())

    df_vectorized = dataframe \
        .withColumn("features_vector", udf_extract_features("image.data")) \
        .withColumn("min", udf_function_min("image.data")) \
        .withColumn("max", udf_function_max("image.data")) \
        .withColumn("mean", udf_function_mean("image.data")) \
        .withColumn("standard_deviation", udf_function_standard_deviation("image.data")) \
        .select(["label", "min", "max", "mean", "standard_deviation"])

    # Assembling features into a vector column
    assembler = VectorAssembler(
        inputCols=["min", "max", "mean", "standard_deviation"],
        outputCol="features"
    )

    df_assembled = assembler \
        .transform(df_vectorized) \
        .select(['features', 'label'])
    df_assembled.cache()

    # Split the dataset into train/test subgroups
    (training_data, test_data) = df_assembled.randomSplit([0.9, 0.1])

    print("Training Dataset Count: " + str(training_data.count()))
    print("Test Dataset Count: " + str(test_data.count()))

    # Random Forest classifier
    rf = RandomForestClassifier(labelCol='label', featuresCol='features', maxDepth=10)
    rf_model = rf.fit(training_data)
    rf_predictions = rf_model.transform(test_data)

    # Evaluation
    multi_evaluator = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
    rf_accuracy = multi_evaluator.evaluate(rf_predictions)
    print('Random Forest classifier Accuracy:', rf_accuracy)

    # Confusion matrix
    df_metrics = rf_predictions \
        .select(['prediction', 'label']) \
        .withColumn('label', func.col('label').cast(FloatType())) \
        .orderBy('prediction')

    df_metrics = df_metrics.select(['prediction', 'label'])
    df_confusion_matrix = MulticlassMetrics(df_metrics.rdd.map(tuple)).confusionMatrix()

    df_efficacy = spark.sparkContext.parallelize(
        [{
            "accuracy": rf_accuracy,
            "matrix": df_confusion_matrix.toArray().tolist()
        }]
    )

    return spark.createDataFrame(df_efficacy)


def mean_value(arr):
    return float(np.mean(arr))


def min_value(arr):
    return float(np.min(arr))


def max_value(arr):
    return float(np.max(arr))


def standard_deviation_value(arr):
    return float(np.std(arr))


def features_vector(arr):
    return DenseVector(np.array(arr))


def hdfs_origin(image):
    if image.origin.startswith("hdfs://"):
        return image.origin
    else:
        return image.origin[7:]


def hdfs_image_data(image):
    return image.data


def classify(descriptor):
    if descriptor == DESCRIPTOR_NORMAL:
        return CLASSNAME_NORMAL
    elif descriptor == DESCRIPTOR_COVID19:
        return CLASSNAME_COVID19
    elif descriptor == DESCRIPTOR_LUNG_OPACITY:
        return CLASSNAME_LUNG_OPACITY
    elif descriptor == DESCRIPTOR_VIRAL_PNEUMONIA:
        return CLASSNAME_VIRAL_PNEUMONIA
    else:
        return CLASSNAME_INVALID


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

CLASSNAME_NORMAL = "Normal"
CLASSNAME_COVID19 = "COVID"
CLASSNAME_LUNG_OPACITY = "Lung_Opacity"
CLASSNAME_VIRAL_PNEUMONIA = "Viral_Pneumonia"
CLASSNAME_INVALID = ''

if __name__ == '__main__':
    main()
