import sys
import numpy as np
import pandas as pd

from functools import reduce
import pyspark.sql.functions as func
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import FloatType, StringType, ArrayType, ByteType

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.linalg import VectorUDT, DenseVector

from dependencies.keras import model_efficacy, add_conv2d_entry_layer, add_conv2d_layer, add_average_pooling2d_layer, \
    add_exit_layers, train_generator_from_dataframe, test_generator_from_dataframe
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import tensorflow as tf

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='radiography_analysis',
        files=['configs/radiography_analysis_config.json']
    )

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

    # ML classification (distributed)
    data_transformed = transform_ml_classification(data_initial, spark)
    load_data(data_transformed, "ml_classification")

    # # DL model compiling/training (not distributed)
    # [data_transformed_matrix, data_transformed_acc] = transform_dl_classification(data_initial, spark)
    # load_data(data_transformed_matrix, "dl_classification_matrix")
    # load_data(data_transformed_acc, "dl_classification_accuracy")
    #
    # # DL model inference (distributed)
    # data_transformed = transform_dl_model_inference(data_initial)
    # load_data(data_transformed, "dl_inference")

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


def transform_dl_classification(dataframe, spark):
    classes = [CLASSNAME_NORMAL, CLASSNAME_COVID19, CLASSNAME_LUNG_OPACITY, CLASSNAME_VIRAL_PNEUMONIA]
    batch_size = 16
    epochs = 12

    udf_function_get_hdfs_origin = udf(hdfs_origin, StringType())
    udf_function_classify = udf(classify, StringType())

    # Preparing the distributed dataframe
    dataframe_keras = dataframe.withColumn("height", dataframe.image.height) \
        .withColumn("width", dataframe.image.width) \
        .withColumn("n_channels", dataframe.image.nChannels) \
        .withColumn("class_name", udf_function_classify("label")) \
        .withColumn("origin", udf_function_get_hdfs_origin("image"))

    dataframe_keras = dataframe_keras.filter(func.col("class_name") != CLASSNAME_INVALID)
    dataframe_keras = dataframe_keras.drop("image", "label")
    dataframe_keras.cache()

    dataframe_keras_master = dataframe_keras.toPandas()

    # Data generators
    # Based on distributed dataframe, batch_size and classes to predict
    [train_datagen, train_gen] = train_generator_from_dataframe(dataframe_keras_master, batch_size, classes)
    [test_datagen, test_gen] = test_generator_from_dataframe(dataframe_keras_master, batch_size, classes)

    # Constructing the deep CNN network
    model = Sequential()
    add_conv2d_entry_layer(model)
    add_conv2d_layer(model)
    add_average_pooling2d_layer(model)
    add_conv2d_layer(model)
    add_conv2d_layer(model)
    add_average_pooling2d_layer(model)
    add_conv2d_layer(model)
    add_average_pooling2d_layer(model)
    add_conv2d_layer(model)
    add_conv2d_layer(model)
    add_average_pooling2d_layer(model)
    add_exit_layers(model)

    # Compiling the model and initiating training
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(learning_rate=0.01, momentum=0.8),
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
        steps_per_epoch=len(train_gen) // batch_size,
        validation_steps=len(test_gen) // batch_size,
        validation_data=test_gen,
        epochs=epochs
    )

    predictions_y = model.predict(test_gen)
    [conf_matrix, accuracy] = model_efficacy(predictions_y, test_gen, classes)

    model.save('./outputs/model/')

    return [
        spark.createDataFrame(conf_matrix),
        spark.createDataFrame(accuracy)
    ]


def transform_dl_model_inference(dataframe):
    num_sample_images = 100
    udf_function_get_hdfs_image = udf(hdfs_image_data, ArrayType(ByteType()))

    # Take random 100 sample images
    dataframe_pred = dataframe \
        .withColumn("image_data", udf_function_get_hdfs_image("image")) \
        .drop("image", "label") \
        .limit(num_sample_images)

    dataframe_pred.cache()

    # Distributed model inference
    # Predict batches across different partitions in parallel
    dataframe_inference = dataframe_pred \
        .select(predict_batch_udf(func.col("image_data")).alias("prediction"))

    return dataframe_inference


@pandas_udf(ArrayType(FloatType()))
def predict_batch_udf(image_batch_iter):
    batch_size = 64
    model = load_model("./outputs/model/")

    for image_batch in image_batch_iter:
        images = np.vstack(image_batch)
        ds_tensors = tf.data.Dataset.from_tensor_slices(images)
        ds_tensors = ds_tensors.map(parse_image, num_parallel_calls=8) \
            .prefetch(5000).batch(batch_size)

        predictions = model.predict(ds_tensors)

        yield pd.Series(list(predictions))


def parse_image(image_data):
    image = tf.image.convert_image_dtype(image_data, dtype=tf.float32) * (2. / 255) - 1
    image = tf.reshape(image, [299, 299, 1])

    return image


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
