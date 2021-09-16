from tempo.tsdf import TSDF
import pyspark.sql.functions as F
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from prophet import Prophet
import datetime
from pyspark.sql.types import *

def calc_anomalies(spark, yaml_file):

    import os
    import yaml

    yaml_path = ''

    if (yaml_file.startswith("s3")):
        import boto3

        s3 = boto3.client('s3')
        s3_bucket = yaml_file.split("s3://")[1]
        bucket_file_tuple = s3_bucket.split("/")
        bucket_name_only = bucket_file_tuple[0]
        file_only = "/".join(bucket_file_tuple[1:])

        s3.download_file(bucket_name_only, file_only, 'ad.yaml')
        yaml_path = '/databricks/driver/ad.yaml'
    elif yaml_file.startswith("dbfs:/") & (os.getenv('DATABRICKS_RUNTIME_VERSION') != None):
        new_dbfs_path = "/" + yaml_file.replace(":", "")
        yaml_path = new_dbfs_path
    else:
        yaml_path = yaml_file

    with open(yaml_path) as f:

        data = yaml.load(f, Loader=yaml.FullLoader)

    import json
    for d in data.keys():
        database = data[d]['database']
        table = data[d]['database'] + '.' + data[d]['name']
        tgt_table = database + '.' + data[d]['name']
        df = spark.table(table)
        partition_cols = data[d]['partition_cols']
        ts_col = data[d]['ts_col']
        mode = data[d]['mode']
        metrics = data[d]['metrics']
        lkbck_window = data[d]['lookback_window']

        # logic to stack metrics instead of individual columns
        l = []
        sep = ', '
        sql_list = sep.join(metrics).split(",")
        n = len(metrics)
        for a in range(n):
            l.append("'{}'".format(metrics[a]) + "," + sql_list[a])
        # partition_col string
        partition_cols_str = ", ".join(partition_cols)
        metrics_cols_str = ", ".join(metrics)
        k = sep.join(l)
        for metric_col in metrics:
            df = df.withColumn(metric_col, F.col(metric_col).cast("double"))

        df.createOrReplaceTempView("tsdf_view")
        stacked = spark.sql("select {}, {}, {}, stack({}, {}) as (metric, value) from tsdf_view".format(ts_col, partition_cols_str, metrics_cols_str, n, k)).withColumn("ts", F.col(ts_col).cast("timestamp"))



        part_cols_w_metrics = partition_cols + metrics

        tsdf = TSDF(stacked, partition_cols = part_cols_w_metrics, ts_col = ts_col)
        moving_avg = tsdf.withRangeStats(['value'], rangeBackWindowSecs=int(lkbck_window)).df
        anomalies = moving_avg.select(ts_col, *partition_cols, 'metric', 'value', 'zscore_' + 'value').withColumn("class_1_anomaly_fl", F.when(F.col('zscore_' + 'value') > 2.5, 1).otherwise(0))


        def forecast_model(pdf: pd.DataFrame) -> pd.DataFrame:
          '''
          Input: a pandas dataframe created from stacekd Spark DataFrame with schema:
                  ts        datetime64[ns]
                  winner            object
                  metric            object
                  value            float64

            Output: a pandas dataframe forecast with schema:
                    ds            datetime64[ns]
                    winner                object
                    metric                object
                    y                    float64
                    yhat                 float64
                    yhat_lower           float64
                    yhat_upper           float64
            '''

          def create_prophet_df(pdf):
            '''
           Prophet only takes data as a dataframe with a ds (datestamp)
           And y (value we want to forecast) column
            '''

            pdf['ds'] = pdf['ts']
            pdf['y'] = pdf['value']
            return pdf[['ds','y']]

          def model_fit(prophet_df):
            model_df = prophet_df.copy()
            m = Prophet(interval_width=0.90, daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=0.01)
            # model fit
            m.fit(model_df)
            return m

          def future_pred(prophet_df, model, periods):
            '''
        returen a pandas dataframe forecast: will be 24 rows longer than input pdf
        '''
            # predict 24 hours, frequency = hourly
            future = model.make_future_dataframe(periods=periods, freq='H')
            forecast = model.predict(future)
            forecast['y'] =  prophet_df['y']

            return forecast[['ds','y', 'yhat', 'yhat_lower', 'yhat_upper']]

          prophet_df = create_prophet_df(pdf).copy()
          prophet_model = model_fit(prophet_df)
          forecast = future_pred(prophet_df, model=prophet_model, periods=24)

          # forecast dataframe schema change: added winner and metric string; avoid null
          metric = pdf['metric'][0]
          forecast['metric'] =  metric
          for col in partition_cols:
            winner = pdf[col][0]
            forecast[col] = winner

          return forecast


        #schema matches the output of func forecast_model forecast pandas dataframe
        customSchema = StructType([StructField("ds", TimestampType())] +
            [StructField(col, StringType()) for col in partition_cols] + [
          StructField("metric", StringType()),
          StructField("y", FloatType()),
          StructField("yhat", FloatType()),
          StructField("yhat_lower", FloatType()),
          StructField("yhat_upper", FloatType())
          ])

        ## Apply Pandas_udf on Spark Dataframe
        forecast_result = stacked.groupBy(*partition_cols).applyInPandas(forecast_model, schema=customSchema)
        forecast_result.createOrReplaceTempView('forecast_result')

        selected_cols = ['ds'] + partition_cols + ['metric', 'yhat', 'y', 'yhat_lower', 'yhat_upper', ]
        f_anomalies = forecast_result.select(*selected_cols).withColumn("class_2_anomaly_fl", F.when( ( (F.col("y") >= F.col("yhat_upper")) | (F.col("y") <= F.col("yhat_lower"))) & (F.col("y").isNotNull()), 1).otherwise( F.when( (F.col("y").isNull()) & ( (F.col("yhat") >= F.col("yhat_upper")) | (F.col("yhat") <= F.col("yhat_lower"))), 1).otherwise(0))).withColumnRenamed("ds", ts_col)

        join_cols = partition_cols + [ts_col] +  ['metric']
        cons_anomalies = anomalies.join(f_anomalies, join_cols).withColumn("anomaly_fl", 0.5*F.col("class_1_anomaly_fl") + 0.5*F.col("class_2_anomaly_fl"))

        # class 1 - 2.5 standard deviations outside mean
        # brand new table
        if mode == 'new':
            spark.sql("create database if not exists {}".format(database))
            cons_anomalies.write.mode('overwrite').option("overwriteSchema", "true").format("delta").saveAsTable(tgt_table + "_tempo_anomaly")
        # append to existing table without DLT
        elif mode == 'append':
            # incremental append with DLT
            print('append for DLT')
        elif (mode == 'incremental') & (os.getenv('DATABRICKS_RUNTIME_VERSION') != None):
            import dlt
            @dlt.view
            def taxi_raw():
              return spark.read.json("/databricks-datasets/nyctaxi/sample/json/")

            # Use the function name as the table name
            @dlt.table
            def filtered_data():
              return dlt.read("taxi_raw").where(...)

            # Use the name parameter as the table name
            @dlt.table(name="filtered_data")
            def create_filtered_data():
              return dlt.read("taxi_raw").where(...)
            #anomalies.write.format("delta").saveAsTable("class1")

