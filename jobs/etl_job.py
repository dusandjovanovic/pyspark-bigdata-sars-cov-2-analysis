from pyspark.sql import Row
from pyspark.sql.functions import col, concat_ws, lit

from dependencies.spark import start_spark


def main():
    spark, log, config = start_spark(
        app_name='my_etl_job',
        files=['configs/etl_config.json'])

    log.warn('etl_job is up-and-running')

    # execute ETL pipeline
    data = extract_data(spark)
    data_transformed = transform_data(data, config['steps_per_floor'])
    load_data(data_transformed)

    log.warn('test_etl_job is finished')
    spark.stop()
    return None


def extract_data(spark):
    df = (
        spark
            .read
            .parquet('tests/test_data/employees'))

    return df


def transform_data(df, steps_per_floor_):
    df_transformed = (
        df
            .select(
            col('id'),
            concat_ws(
                ' ',
                col('first_name'),
                col('second_name')).alias('name'),
            (col('floor') * lit(steps_per_floor_)).alias('steps_to_desk')))

    return df_transformed


def load_data(df):
    (df
     .coalesce(1)
     .write
     .csv('loaded_data', mode='overwrite', header=True))
    return None


if __name__ == '__main__':
    main()
