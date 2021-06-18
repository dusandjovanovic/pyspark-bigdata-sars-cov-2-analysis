import sys

import numpy as np
import pandas as pd
import pyspark.sql.functions as func
from pyspark.sql.functions import udf
from functools import reduce
from pyspark.sql.types import FloatType, StringType

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.linalg import VectorUDT, DenseVector

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D
from keras.layers import Conv2D, Activation
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

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

    # # ML classification (distributed)
    # data_transformed = transform_ml_classification(data_initial, spark)
    # load_data(data_transformed, "ml_classification")

    # DL classification (not distributed)
    [data_transformed_matrix, data_transformed_acc] = transform_dl_classification(data_initial, spark)
    load_data(data_transformed_matrix, "dl_classification_matrix")
    load_data(data_transformed_acc, "dl_classification_accuracy")

    # DL distributed model execution
    # data_transformed = transform_dl_model_execution(data_initial, spark)
    # load_data(data_transformed, "dl_predictions")

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

    dataframe_repartitioned = dataframe_merged.repartition(200)

    return dataframe_repartitioned


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

    df_efficacy = spark.sparkContext.parallelize([{
        "accuracy": rf_accuracy,
        "matrix": df_confusion_matrix.toArray().tolist()
    }])

    return spark.createDataFrame(df_efficacy)


def transform_dl_classification(dataframe, spark):
    classes = [CLASSNAME_NORMAL, CLASSNAME_COVID19, CLASSNAME_LUNG_OPACITY, CLASSNAME_VIRAL_PNEUMONIA]
    batch_size = 16
    epochs = 1

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

    # Constructing the model
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

    model.save('./outputs/models/')

    return [
        spark.createDataFrame(conf_matrix),
        spark.createDataFrame(accuracy)
    ]


def transform_dl_model_execution(dataframe, spark):
    num_partitions = 10
    udf_function_get_hdfs_origin = udf(hdfs_origin, StringType())
    udf_function_classify = udf(classify, StringType())

    dataframe_pred = dataframe.withColumn("class_name", udf_function_classify("label")) \
        .withColumn("origin", udf_function_get_hdfs_origin("image")) \
        .drop("image", "label")

    rdd_pred = dataframe_pred \
        .rdd \
        .repartition(num_partitions) \
        .mapPartitions(predict_for_partition)

    return spark.createDataFrame(rdd_pred)


def predict_for_partition(partition):
    model = load_model("./outputs/models")

    for row in partition:
        prediction = model.predict_classes(row)
        yield prediction


def model_efficacy(predictions_y, test_gen, classes):
    predictions = np.array(list(map(lambda x: np.argmax(x), predictions_y)))

    conf_matrix = pd.DataFrame(
        confusion_matrix(test_gen.classes, predictions),
        columns=classes,
        index=classes
    )

    acc = accuracy_score(test_gen.classes, predictions)

    results_all = precision_recall_fscore_support(
        test_gen.classes, predictions,
        average='macro',
        zero_division=1
    )
    results_class = precision_recall_fscore_support(
        test_gen.classes,
        predictions,
        average=None, zero_division=1
    )

    # Organise the Results into a Dataframe
    metric_columns = ['Precision', 'Recall', 'F-Score', 'S']
    accuracy = pd.concat([pd.DataFrame(list(results_class)).T, pd.DataFrame(list(results_all)).T])
    accuracy.columns = metric_columns
    accuracy.index = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia', 'Total']

    return [conf_matrix, accuracy]


def add_conv2d_entry_layer(model):
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=(299, 299, 1)))
    model.add(BatchNormalization())
    return None


def add_conv2d_layer(model):
    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(BatchNormalization())
    return None


def add_average_pooling2d_layer(model):
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    return None


def add_exit_layers(model):
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))
    return None


def train_generator_from_dataframe(dataframe_keras_master, batch_size, classes):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=dataframe_keras_master,
        directory=None,
        x_col="origin",
        y_col="class_name",
        classes=classes,
        batch_size=batch_size,
        target_size=(299, 299),
        class_mode="categorical",
        subset="training",
        color_mode="grayscale",
        shuffle=True,
        validate_filenames=False
    )

    return [train_datagen, train_gen]


def test_generator_from_dataframe(dataframe_keras_master, batch_size, classes):
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=dataframe_keras_master,
        directory=None,
        x_col="origin",
        y_col="class_name",
        classes=classes,
        batch_size=batch_size,
        target_size=(299, 299),
        class_mode="categorical",
        subset="validation",
        color_mode="grayscale",
        shuffle=False,
        validate_filenames=False
    )

    return [test_datagen, test_gen]


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


def features_vector(arr):
    return DenseVector(np.array(arr))


def hdfs_origin(image):
    if image.origin.startswith("hdfs://"):
        return image.origin
    else:
        return image.origin[7:]


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
