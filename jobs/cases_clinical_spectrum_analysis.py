from pyspark.sql.types import IntegerType, StringType, DoubleType
import pyspark.sql.functions as func
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualisation.dependencies import color_scheme
import sys

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
    data_transformed = transform_data(data, sql_context)

    # hemoglobin/eosinophils values analysis
    data_transformed = transform_hemoglobin_red_blood_cells_values(data_transformed)
    load_data(data_transformed, "hemoglobin_red_blood_cells_values")

    # average age/result aggregated distribution analysis
    data_transformed = transform_aggregate(data_transformed, sql_context)
    load_data(data_transformed, "age_result_distribution")

    # age relations to test outcomes
    data_transformed = transform_age_relations(data_transformed, sql_context)
    load_data(data_transformed, "age_relations")

    # care type admition relations of positive patients
    data_transformed = transform_care_relations(data_transformed, sql_context)
    load_data(data_transformed, "care_relations")

    # predictions
    data_transformed = transform_predictions(data_transformed, sql_context)
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


def transform_hemoglobin_red_blood_cells_values(dataframe):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    df_hemoglobin = dataframe_clean.select("Hemoglobin").withColumn("Hemoglobin",
                                                                    func.round(dataframe_clean["Hemoglobin"],
                                                                               2)).toPandas()
    df_eosinophils = dataframe_clean.select("Red blood Cells").withColumn("Red blood Cells",
                                                                          func.round(dataframe_clean["Red blood Cells"],
                                                                                     2)).toPandas()

    fig = px.histogram(df_hemoglobin, x="Hemoglobin", title="Hemoglobin distribution",
                       color_discrete_sequence=[color_scheme.color_500], opacity=0.8, marginal="rug")
    fig.show()

    fig = px.histogram(df_eosinophils, x="Red blood Cells", title="Red blood Cells distribution",
                       color_discrete_sequence=[color_scheme.color_300], opacity=0.8, marginal="rug")
    fig.show()

    return dataframe


def transform_aggregate(dataframe, sql_context):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    df_age_select = dataframe_clean.select(func.col("SARS-Cov-2 exam result").alias("result"),
                                           func.col('Patient age quantile').alias('age'))

    df_age_select.write.mode('overwrite').option("header", "true").save("temporary.parquet",
                                                                        format="parquet")

    df_sql = sql_context.sql("SELECT * FROM parquet.`./temporary.parquet`")
    df_aggregate = df_sql.groupBy("result").agg(func.max("age"), func.avg("age")).toPandas()
    fig = px.line(df_aggregate, x="result", y="avg(age)", title="Average age/result distribution",
                  log_y=True, color_discrete_sequence=[color_scheme.color_400])
    fig.show()

    return dataframe


def transform_age_relations(dataframe, sql_context):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    udf_function_positive = func.udf(is_positive, StringType())
    udf_function_negative = func.udf(is_negative, StringType())

    df_age = dataframe_clean.select(func.col("SARS-Cov-2 exam result").alias("result"),
                                    func.col('Patient age quantile').alias('age'))

    df_age_positive = df_age.withColumn("positive", udf_function_positive("result"))
    df_age_negative = df_age.withColumn("negative", udf_function_negative("result"))

    display_positive = df_age_positive.select("positive").toPandas()["positive"]
    display_negative = df_age_negative.select("negative").toPandas()["negative"]
    display_age = df_age.select("age").toPandas()["age"]

    rec_age_fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Age and Positive/Negative correlation", "Positive/Negative"))
    rec_age_fig.add_trace(go.Box(x=display_positive, y=display_age, name="Positive"), row=1, col=1)
    rec_age_fig.add_trace(go.Box(x=display_negative, y=display_age, name="Negative"), row=1, col=2)
    rec_age_fig.update_traces(boxpoints='all')
    rec_age_fig.update_layout(title_text="Subplots of age in relation a positive/negative test result")
    rec_age_fig.show()

    return dataframe


def transform_care_relations(dataframe, sql_context):
    dataframe_clean = dataframe.fillna(0)
    dataframe_clean = dataframe_clean.replace("nan", "0")

    udf_function_to_numeric = func.udf(negative_positive_to_numeric, IntegerType())

    df_transformed_numeric = dataframe_clean.withColumn("result", udf_function_to_numeric("SARS-Cov-2 exam result"))
    df_transformed_positive = df_transformed_numeric.filter(df_transformed_numeric.result == 1)
    df_transformed_positive_display = df_transformed_positive.toPandas()

    fig = px.bar(df_transformed_positive_display, y="result", x="Patient addmited to regular ward (1=yes, 0=no)",
                 color_discrete_sequence=[color_scheme.color_400, color_scheme.color_500],
                 title="Positive patients admited to regular care")
    fig.show()

    fig_intensive = px.bar(df_transformed_positive_display, y="result",
                           x="Patient addmited to intensive care unit (1=yes, 0=no)",
                           color_discrete_sequence=[color_scheme.color_900, color_scheme.color_500],
                           title="Positive patients admited to intensive care")
    fig_intensive.show()

    return dataframe


def transform_predictions(dataframe, sql_context):
    df_transformed = dataframe.drop("Patient addmited to regular ward (1=yes, 0=no)",
                                    "Patient addmited to semi-intensive unit (1=yes, 0=no)",
                                    "Patient addmited to intensive care unit (1=yes, 0=no)")

    show_predictions_missing_values(df_transformed)

    df_transformed = df_transformed.drop("Mycoplasma pneumoniae", "Urine - Sugar",
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

    show_predictions_value_distribution(df_transformed)
    show_predictions_test_result_distribution(df_transformed)

    # build the dataset to be used as a rf_model base
    outcome_features = ["SARS-Cov-2 exam result"]
    required_features = ['Hemoglobin', 'Hematocrit', 'Platelets', 'Eosinophils', 'Red blood Cells', 'Lymphocytes',
                         'Leukocytes', 'Basophils', 'Monocytes']

    df_transformed_model = df_transformed.select(required_features + outcome_features)

    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    model_data = assembler.transform(df_transformed_model)
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

    show_predictions_accuracy_distribution(rf_accuracy, dt_accuracy, lr_accuracy, gb_accuracy)

    return dataframe


def show_predictions_missing_values(dataframe):
    df_transformed_null = dataframe.select(
        [func.count(func.when(func.isnan(c) | func.isnull(c), c)).alias(c) for (c, c_type) in
         dataframe.dtypes])

    df_null = df_transformed_null.toPandas()
    df_null = df_null.rename(index={0: 'count'}).T.sort_values("count", ascending=False)

    fig = px.bar(df_null, y="count",
                 color_discrete_sequence=[color_scheme.color_400, color_scheme.color_500],
                 title="Statistics of missing (null/nan) values across columns")
    fig.show()

    return None


def show_predictions_value_distribution(dataframe):
    df_results = dataframe.select("SARS-Cov-2 exam result").toPandas()
    df_hemoglobin = dataframe.select("Hemoglobin").toPandas()
    df_hematocrit = dataframe.select("Hematocrit").toPandas()
    df_plateletst = dataframe.select("Platelets").toPandas()
    df_eosinophils = dataframe.select("Eosinophils").toPandas()
    df_red_blood_cells = dataframe.select("Red blood Cells").toPandas()
    df_lymphocytes = dataframe.select("Lymphocytes").toPandas()
    df_leukocytes = dataframe.select("Leukocytes").toPandas()
    df_basophils = dataframe.select("Basophils").toPandas()
    df_monocytes = dataframe.select("Monocytes").toPandas()

    fig = make_subplots(rows=3, cols=3, subplot_titles=(
        "Hemoglobin/Exam Result", "Platelets/Exam Result", "Eosinophils/Exam Result", "Red blood Cells/Exam Result",
        "Lymphocytes/Exam Result", "Leukocytes/Exam Result", "Basophils/Exam Result", "Monocytes/Exam Result",
        "Hematocrit/Exam Result"))

    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_hemoglobin['Hemoglobin'], mode='markers',
                   marker=dict(color=color_scheme.color_900)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_plateletst['Platelets'], mode='markers',
                   marker=dict(color=color_scheme.color_800)), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_eosinophils['Eosinophils'], mode='markers',
                   marker=dict(color=color_scheme.color_700)), row=1, col=3)
    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_red_blood_cells['Red blood Cells'], mode='markers',
                   marker=dict(color=color_scheme.color_600)), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_lymphocytes['Lymphocytes'], mode='markers',
                   marker=dict(color=color_scheme.color_500)), row=2, col=2)
    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_leukocytes['Leukocytes'], mode='markers',
                   marker=dict(color=color_scheme.color_400)), row=2, col=3)
    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_basophils['Basophils'], mode='markers',
                   marker=dict(color=color_scheme.color_300)), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_monocytes['Monocytes'], mode='markers',
                   marker=dict(color=color_scheme.color_200)), row=3, col=2)
    fig.add_trace(
        go.Scatter(x=df_results['SARS-Cov-2 exam result'], y=df_hematocrit['Hematocrit'], mode='markers',
                   marker=dict(color=color_scheme.color_100)), row=3, col=3)

    fig.show()

    return None


def show_predictions_test_result_distribution(dataframe):
    udf_function_result = func.udf(transform_result, StringType())

    df_transformed = dataframe.withColumn("result", udf_function_result("SARS-Cov-2 exam result"))
    df_transformed_collected = df_transformed.groupBy('result').count().toPandas()

    fig = px.pie(df_transformed_collected, values='count', names='result',
                 title="Statistics of test result distribution",
                 color_discrete_sequence=[color_scheme.color_100, color_scheme.color_400])
    fig.show()

    return None


def show_predictions_accuracy_distribution(rf_accuracy, dt_accuracy, lr_accuracy, gb_accuracy):
    fig = go.Figure(data=[go.Bar(y=[rf_accuracy, dt_accuracy, lr_accuracy, gb_accuracy],
                                 x=['Random Forest classifier Accuracy', 'Decision Tree Accuracy',
                                    'Logistic Regression Accuracy', 'Gradient-boosted Trees Accuracy'])])
    fig.update_traces(marker_color=color_scheme.color_200, marker_line_color=color_scheme.color_600,
                      marker_line_width=1.5, opacity=0.6)
    fig.update_layout(title_text='Comparison of classifier accuracy reports')
    fig.show()

    return None


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
     .csv("./outputs/cases_clinical_spectrum_analysis/" + name, mode='overwrite', header=True))
    return None


if __name__ == '__main__':
    main()
