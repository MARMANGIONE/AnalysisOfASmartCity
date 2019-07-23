# -*- coding: utf-8  -*-

from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from scipy.stats import mstats
import numpy as np
from pyspark.sql.types import *


spark = SparkSession\
    .builder\
    .enableHiveSupport()\
    .getOrCreate()\

# ------------- KRUSKAL TEST -------------------------------------------------------------------------------------------
def kruskalForWorkingDays(november):
    november = november.withColumn('internet_level', func.when((func.col('internet') >= 0) & (func.col('internet') < 40), 0)\
                   .when((func.col('internet') >= 40) & (func.col('internet') < 50),1)\
                   .when((func.col('internet') >= 50) & (func.col('internet') < 60),2)\
                   .otherwise(3))
    november = november.orderBy('date_istant')
    november.show()
    kruskalTest(november)

def kruskalForWeekend(november):
    november = november.withColumn('internet_level', func.when((func.col('internet') >= 0) & (func.col('internet') < 40), 0)\
                   .when((func.col('internet') >= 40) & (func.col('internet') < 50),1)\
                   .when((func.col('internet') >= 50) & (func.col('internet') < 60),2)\
                   .otherwise(3))
    november = november.orderBy('date_istant')
    november.show()
    kruskalTest(november)

def kruskalTest(november):
    Col_1 = np.concatenate(november.select('rain_intensity').collect(), axis=0)
    print(Col_1)
    Col_2 = np.concatenate(november.select('internet_level').collect(), axis=0)
    print("Kruskal Wallis H-test test:")
    H, pval = mstats.kruskalwallis(Col_1, Col_2)
    print("H-statistic:", H)
    print("P-Value:", pval)
    if pval < 0.05:
        print("Reject NULL hypothesis - Significant differences exist between groups.")
    if pval > 0.05:
        print("Accept NULL hypothesis - No significant difference between groups.")
#-----------------------------------------------------------------------------------------------------------------------

#[1] Telecommunication data ingestion
cdr = spark.read.option("delimiter", "\t")\
  .csv('inputs/telecom/november/*.txt')\
  .toDF('square_id', 'time_interval', 'country_code', 'SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'internet_traffic')
cdr = cdr.withColumn('time_istant', func.from_unixtime(cdr.time_interval/1000, 'yyyy-MM-dd HH:mm:ss:SSS"'))
cdr = cdr.withColumn('time_istant', func.unix_timestamp(cdr.time_istant, 'yyyy-MM-dd HH:mm:ss:SSS"').cast('timestamp'))
cdr = cdr.withColumn('weekday', func.dayofweek(cdr['time_istant']))
cdr = cdr.withColumn('time_istant', func.date_trunc('hour','time_istant'))
cdr = cdr.withColumn("time_istant", func.from_unixtime(func.unix_timestamp(cdr.time_istant), "yyyy/MM/dd HH:mm"))
cdr = cdr.withColumn('SMS_in', cdr['SMS_in'].cast(FloatType()))
cdr = cdr.withColumn('SMS_out', cdr['SMS_out'].cast(FloatType()))
cdr = cdr.withColumn('Call_in', cdr['Call_in'].cast(FloatType()))
cdr = cdr.withColumn('Call_out', cdr['Call_out'].cast(FloatType()))
cdr= cdr.groupBy(cdr.time_istant,cdr.weekday)\
  .agg(func.sum("SMS_out").alias("smsInUscita"),func.sum("SMS_in").alias("smsInEntrata"),func.sum("Call_in").alias("chiamateInEntrata"),func.sum("Call_out").alias("chiamateInUscita"),func.sum("internet_traffic").alias("internet"))
cdr = cdr.withColumn('SMS_total', cdr['smsInUscita'] + cdr['smsInEntrata'])
cdr = cdr.withColumn('Call_total', cdr['chiamateInEntrata'] + cdr['chiamateInUscita'])
cdr.createOrReplaceTempView("telecommunicationData")
mobileDF = spark.sql("select SMS_total, Call_total, internet, time_istant, weekday from telecommunicationData ")
mobileDF.show()

#[2] Weather data ingestion
sensors = spark.read.option("delimiter", ",")\
  .csv('inputs/mi_meteo_legend.csv')\
  .toDF('sensor_id','street_name','lat', 'lon','sensor_type','unity_of_measure')
sensors = sensors.filter(sensors['sensor_type'] == "Precipitation")
sensors = sensors.filter(sensors['street_name'] == "Milano - via Lambrate")

weather = spark.read.option("delimiter", ",")\
   .csv('inputs/weather_phenomena/*.csv')\
  .toDF('sensor_id','time_istant','measurement')


#[3] Grouping by date abd Precipitation intensity classification
ws = weather.join(sensors, on='sensor_id', how='left')
ws = ws.withColumn('rain_intensity', func.when(func.col('measurement') == 0, 0)\
                   .when((func.col('measurement') > 0) & (func.col('measurement') < 2.6),1)\
                   .when((func.col('measurement') >= 2.6) & (func.col('measurement') < 7.6),2)\
                   .otherwise(3))
ws = ws.filter(ws['rain_intensity'] > 0)
city = mobileDF.join(ws,on='time_istant', how='left')
city.show()
city = city.withColumn('date_again', func.from_unixtime(func.unix_timestamp(city.time_istant, 'yyyy/MM/dd HH:mm'),'yyyy-MM-dd HH:mm'))
city = city.filter(city.sensor_type.isNotNull())

#[4] Filtering only for working hours
city.createOrReplaceTempView("sparkCityData")
novemberactivity = city
novemberactivity = novemberactivity.withColumn('date_istant', func.unix_timestamp(novemberactivity.time_istant, 'yyyy/MM/dd HH:mm').cast('timestamp'))
novemberactivity = novemberactivity.withColumn('hour', func.hour(novemberactivity['date_istant']))
novemberactivity = novemberactivity.withColumn('hour', novemberactivity['hour'].cast(FloatType()))
novemberactivity = novemberactivity.filter(novemberactivity['hour'] > 7)
novemberactivity = novemberactivity.filter(novemberactivity['hour'] < 20)
novemberactivity.createOrReplaceTempView("novemberView")
november = spark.sql("select date_istant,rain_intensity, SMS_total, Call_total, internet,weekday from novemberView")

#[5] Division for working days and weekend
november = november.withColumn('rain_intensity', november['rain_intensity'].cast(FloatType()))
november = november.withColumn('weekday', november['weekday'].cast(FloatType()))
november = november.withColumn('internet', november['internet'].cast(DecimalType()))
november = november.withColumn('internet',(func.round(november['internet'],1)).cast("string"))
november = november.withColumn('internet', (november['internet']).substr(1,2))
november = november.withColumn('internet', november['internet'].cast(FloatType()))
workingdayIntensity = november.filter((november['weekday'] >=1) & (november['weekday'] <=5))
weekendIntensity = november.filter((november['weekday'] >=6) & (november['weekday'] <=7))
kruskalForWorkingDays(workingdayIntensity)
kruskalForWeekend(weekendIntensity)
