import sys

import pyspark.sql.functions as func
from pyspark.sql.types import IntegerType, StringType, DoubleType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='cases_clinical_spectrum_analysis',
        files=['configs/cases_clinical_spectrum_config.json'])

    log.warn('Running cases_clinical_spectrum analysis...')

    # extracting and transforming the dataset
    data = extract_data(spark)
    data_initial = transform_data(data, sql_context)

    # hemoglobin values analysis
    data_transformed = transform_hemoglobin_values(data_initial)
    load_data(data_transformed, "hemoglobin_values")

    # red blood cells values analysis
    data_transformed = transform_red_blood_cells_values(data_initial)
    load_data(data_transformed, "red_blood_cells_values")

    # average age/result aggregated distribution analysis
    data_transformed = transform_aggregate_age_result(data_initial, sql_context)
    load_data(data_transformed, "aggregate_age_result")

    # age relations to test outcomes
    data_transformed = transform_age_relations(data_initial, sql_context)
    load_data(data_transformed, "age_relations")

    # care type relations of positive patients
    data_transformed = transform_care_relations(data_initial, sql_context)
    load_data(data_transformed, "care_relations")

    # missing values for predictions
    data_transformed = transform_predictions_missing_values(data_initial, sql_context)
    load_data(data_transformed, "predictions_missing_values")

    # value distribution for predictions
    data_transformed = transform_predictions_value_distribution(data_initial, sql_context)
    load_data(data_transformed, "predictions_value_distribution")

    # test result distribution for predictions
    data_transformed = transform_predictions_test_result_distribution(data_initial, sql_context)
    load_data(data_transformed, "predictions_test_result_distribution")

    # predictions
    data_transformed = transform_predictions(data_initial, sql_context, spark)
    load_data(data_transformed, "predictions")

    log.warn('Terminating cases_clinical_spectrum analysis...')

    spark.stop()
    return None


def extract_data(spark):
    dataframe = spark.read.csv(sys.argv[1], header=True, dateFormat="yyyy-MM-dd")

    return dataframe


def transform_data(frame, sql_context):
    dt_transformed = frame

    columns = dt_transformed.columns

    for col_name in columns:
        dt_transformed = dt_transformed.withColumn(col_name, func.col(col_name).cast('string'))

    return dt_transformed


def transform_hemoglobin_values(dataframe):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    df_hemoglobin = dataframe_clean.select("Hemoglobin").withColumn("Hemoglobin",
                                                                    func.round(dataframe_clean["Hemoglobin"],
                                                                               2))

    return df_hemoglobin


def transform_red_blood_cells_values(dataframe):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    df_red_blood_cells = dataframe_clean.select("Red blood Cells").withColumn("Red blood Cells",
                                                                              func.round(
                                                                                  dataframe_clean["Red blood Cells"],
                                                                                  2))

    return df_red_blood_cells


def transform_aggregate_age_result(dataframe, sql_context):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    df_age_select = dataframe_clean.select(func.col("SARS-Cov-2 exam result").alias("result"),
                                           func.col('Patient age quantile').alias('age'))

    df_age_select.write.mode('overwrite').option("header", "true").save("temporary.parquet",
                                                                        format="parquet")

    df_sql = sql_context.sql("SELECT * FROM parquet.`./temporary.parquet`")
    df_aggregate = df_sql.groupBy("result").agg(func.max("age"), func.avg("age"))

    return df_aggregate


def transform_age_relations(dataframe, sql_context):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    udf_function_positive = func.udf(is_positive, StringType())
    udf_function_negative = func.udf(is_negative, StringType())

    df_age = dataframe_clean.select(func.col("SARS-Cov-2 exam result").alias("result"),
                                    func.col('Patient age quantile').alias('age'))

    df_age_with_positive = df_age.withColumn("positive", udf_function_positive("result"))
    df_age_with_positive_negative = df_age_with_positive.withColumn("negative", udf_function_negative("result"))

    return df_age_with_positive_negative


def transform_care_relations(dataframe, sql_context):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    udf_function_to_numeric = func.udf(negative_positive_to_numeric, IntegerType())

    df_transformed_numeric = dataframe_clean.withColumn("result", udf_function_to_numeric("SARS-Cov-2 exam result"))
    df_transformed_positive = df_transformed_numeric.filter(df_transformed_numeric.result == 1)

    return df_transformed_positive


def transform_predictions(dataframe, sql_context, spark):
    df_transformed = dataframe.drop("Patient addmited to regular ward (1=yes, 0=no)",
                                    "Patient addmited to semi-intensive unit (1=yes, 0=no)",
                                    "Patient addmited to intensive care unit (1=yes, 0=no)")

    df_transformed = dismiss_missing_values(df_transformed)

    # build the dataset to be used as a rf_model base
    outcome_features = ["SARS-Cov-2 exam result"]
    required_features = ['Hemoglobin', 'Hematocrit', 'Platelets', 'Eosinophils', 'Red blood Cells', 'Lymphocytes',
                         'Leukocytes', 'Basophils', 'Monocytes']

    df_transformed_model = df_transformed.select(required_features + outcome_features)
    df_transformed_model_sql = sql_context.createDataFrame(df_transformed_model.collect())

    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    model_data = assembler.transform(df_transformed_model_sql)
    model_data.show()

    # split the dataset into train/test subgroups
    (training_data, test_data) = model_data.randomSplit([0.8, 0.2], seed=2020)
    print("Training Dataset Count: " + str(training_data.count()))
    print("Test Dataset Count: " + str(test_data.count()))

    # Random Forest classifier
    rf = RandomForestClassifier(labelCol='SARS-Cov-2 exam result', featuresCol='features', maxDepth=5)
    rf_model = rf.fit(training_data)
    rf_predictions = rf_model.transform(test_data)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol='SARS-Cov-2 exam result', metricName='accuracy')
    rf_accuracy = multi_evaluator.evaluate(rf_predictions)
    print('Random Forest classifier Accuracy:', rf_accuracy)

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(featuresCol='features', labelCol='SARS-Cov-2 exam result', maxDepth=3)
    dt_model = dt.fit(training_data)
    dt_predictions = dt_model.transform(test_data)
    dt_predictions.select(outcome_features + required_features).show(10)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol='SARS-Cov-2 exam result', metricName='accuracy')
    dt_accuracy = multi_evaluator.evaluate(dt_predictions)
    print('Decision Tree Accuracy:', dt_accuracy)

    # Logistic Regression Model
    lr = LogisticRegression(featuresCol='features', labelCol='SARS-Cov-2 exam result', maxIter=10)
    lr_model = lr.fit(training_data)
    lr_predictions = lr_model.transform(test_data)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol='SARS-Cov-2 exam result', metricName='accuracy')
    lr_accuracy = multi_evaluator.evaluate(lr_predictions)
    print('Logistic Regression Accuracy:', lr_accuracy)

    # Gradient-boosted Tree classifier Model
    gb = GBTClassifier(labelCol='SARS-Cov-2 exam result', featuresCol='features')
    gb_model = gb.fit(training_data)
    gb_predictions = gb_model.transform(test_data)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol='SARS-Cov-2 exam result', metricName='accuracy')
    gb_accuracy = multi_evaluator.evaluate(gb_predictions)
    print('Gradient-boosted Trees Accuracy:', gb_accuracy)

    rdd = spark.sparkContext.parallelize([rf_accuracy, dt_accuracy, lr_accuracy, gb_accuracy])
    predictions_dataframe = spark.createDataFrame(rdd, FloatType())

    return predictions_dataframe


def transform_predictions_missing_values(dataframe, sql_context):
    df_transformed_null = dataframe.select(
        [func.count(func.when(func.isnan(c) | func.isnull(c), c)).alias(c) for (c, c_type) in
         dataframe.dtypes])

    return df_transformed_null


def transform_predictions_value_distribution(dataframe, sql_context):
    df_transformed = dismiss_missing_values(dataframe)

    return df_transformed


def transform_predictions_test_result_distribution(dataframe, sql_context):
    df_transformed = dismiss_missing_values(dataframe)
    udf_function_result = func.udf(transform_result, StringType())
    df_transformed = df_transformed.withColumn("result", udf_function_result("SARS-Cov-2 exam result"))
    df_transformed_collected = df_transformed.groupBy('result').count()

    return df_transformed_collected


def dismiss_missing_values(dataframe):
    df_transformed = dataframe.drop("Mycoplasma pneumoniae", "Urine - Sugar",
                                    "Prothrombin time (PT), Activity", "D-Dimer",
                                    "Fio2 (venous blood gas analysis)", "Urine - Nitrite",
                                    "Vitamin B12")

    df_transformed = df_transformed.fillna("0")
    df_transformed = df_transformed.replace("nan", "0")
    df_transformed = df_transformed.na.fill("0")
    columns = [c for c in df_transformed.columns if c not in {'Patient ID'}]

    df_transformed = df_transformed.replace('not_detected', '0')
    df_transformed = df_transformed.replace('detected', '1')
    df_transformed = df_transformed.replace('absent', '0')
    df_transformed = df_transformed.replace('present', '1')
    df_transformed = df_transformed.replace('negative', '0')
    df_transformed = df_transformed.replace('positive', '1')

    for col in columns:
        df_transformed = df_transformed.withColumn(col, df_transformed[col].cast(DoubleType()))

    return df_transformed


def is_positive(value):
    if value == 'positive':
        return '1'
    else:
        return '0'


def is_negative(value):
    if value == 'negative':
        return '1'
    else:
        return '0'


def transform_result(value):
    if value == 0:
        return 'Negative test result'
    else:
        return 'Positive test result'


def negative_positive_to_numeric(value):
    if value == 'negative':
        return 0
    else:
        return 1


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .json("./outputs/cases_clinical_spectrum_analysis/" + name, mode='overwrite'))
    return None


if __name__ == '__main__':
    main()
