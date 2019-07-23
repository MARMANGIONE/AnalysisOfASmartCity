# -*- coding: utf-8  -*-
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.sql.types import *
import milano_poi_zones as pois


#[1] Mobile data ingestion
spark = SparkSession\
    .builder\
    .enableHiveSupport()\
    .getOrCreate()\

cdr = spark.read.option("delimiter", "\t")\
  .csv('inputs/telecom/november/*.txt')\
  .toDF('square_id', 'time_interval', 'country_code', 'SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'internet_traffic')

#[2] Filter of mobile data based on the intensity of the points of interest present for each square_id
lowDensityList = pois.filter(1)
cdr = cdr.filter(cdr.square_id.isin(lowDensityList))

#[3] Group by time_istant
cdr = cdr.withColumn('time_istant', func.from_unixtime(cdr.time_interval/1000, 'yyyy-MM-dd HH:mm:ss:SSS"'))
cdr = cdr.withColumn('time_istant', func.unix_timestamp(cdr.time_istant, 'yyyy-MM-dd HH:mm:ss:SSS"').cast('timestamp'))
cdr = cdr.withColumn('time_istant', func.date_trunc('hour','time_istant'))
cdr = cdr.withColumn("time_istant", func.from_unixtime(func.unix_timestamp(cdr.time_istant), "yyyy/MM/dd HH:mm"))
cdr = cdr.withColumn('SMS_in', cdr['SMS_in'].cast(FloatType()))
cdr = cdr.withColumn('SMS_out', cdr['SMS_out'].cast(FloatType()))
cdr = cdr.withColumn('Call_in', cdr['Call_in'].cast(FloatType()))
cdr = cdr.withColumn('Call_out', cdr['Call_out'].cast(FloatType()))
cdr= cdr.groupBy(cdr.time_istant)\
  .agg(func.sum("SMS_out").alias("smsInUscita"),func.sum("SMS_in").alias("smsInEntrata"),func.sum("Call_in").alias("chiamateInEntrata"),func.sum("Call_out").alias("chiamateInUscita"),func.sum("internet_traffic").alias("internet"))
cdr = cdr.withColumn('SMS_total', cdr['smsInUscita'] + cdr['smsInEntrata'])
cdr = cdr.withColumn('Call_total', cdr['chiamateInEntrata'] + cdr['chiamateInUscita'])
cdr.createOrReplaceTempView("telecommunicationData")
mobileDF = spark.sql("select SMS_total, Call_total, internet, time_istant from telecommunicationData ")

#[4] Weather data ingestion
sensors = spark.read.option("delimiter", ",")\
  .csv('inputs/weather_legend/*.csv')\
  .toDF('sensor_id','street_name','lat', 'lon','sensor_type','unity_of_measure')

weather = spark.read.option("delimiter", ",")\
   .csv('inputs/weather_phenomena/*.csv')\
   .toDF('sensor_id','time_istant','measurement')

ws = weather.join(sensors, on='sensor_id', how='left')

#[5] Join by time_istant
city = mobileDF.join(ws,on='time_istant', how='left')
city = city.withColumn('date_again', func.from_unixtime(func.unix_timestamp(city.time_istant, 'yyyy/MM/dd HH:mm'),'yyyy-MM-dd HH:mm'))


city.createOrReplaceTempView("sparkCityData")

# FIRST ACTIVITIES TO INVESTIGATE THE DATASET
#maximumSmsActivity = spark.sql("select time_istant,measurement,street_name, sensor_type, unity_of_measure, SMS_total from sparkCityData where SMS_total in (select max(SMS_total) from sparkCityData)").show()
#minimumSmsActivity = spark.sql("select time_istant,measurement,street_name, sensor_type, unity_of_measure, SMS_total from sparkCityData where SMS_total in (select min(SMS_total) from sparkCityData)").show()
#maximumCallActivity = spark.sql("select time_istant,measurement,street_name, sensor_type, unity_of_measure, Call_total from sparkCityData where Call_total in (select max(Call_total) from sparkCityData)").show()
#mainimumCallActivity = spark.sql("select time_istant,measurement,street_name, sensor_type, unity_of_measure, Call_total from sparkCityData where Call_total in (select min(Call_total) from sparkCityData)").show()
#maximumInternetActivity = spark.sql("select time_istant,measurement,street_name, sensor_type, unity_of_measure, internet from sparkCityData where internet in (select max(internet) from sparkCityData)").show()
#mainimumInternetActivity = spark.sql("select time_istant,measurement,street_name, sensor_type, unity_of_measure, internet from sparkCityData where internet in (select min(internet) from sparkCityData)").show()

#[6] Filter dataset only for temperature
novemberactivity = city
novemberactivity = novemberactivity.withColumn('date_istant', func.unix_timestamp(novemberactivity.time_istant, 'yyyy/MM/dd HH:mm').cast('timestamp'))
novemberactivity.filter(novemberactivity['street_name'] == "Milano - via Brera")
novemberactivity.createOrReplaceTempView("novemberView")
november = spark.sql("select date_istant,measurement, sensor_type, unity_of_measure, SMS_total, Call_total, internet from novemberView")
november = november.filter(november['sensor_type'] == "Temperature")


november = november.withColumn('measurement', november['measurement'].cast(FloatType()))
november = november.withColumn('SMS_total', november['SMS_total'].cast(FloatType()))
november = november.withColumn('Cal_total', november['Call_total'].cast(FloatType()))
november = november.withColumn('internet', november['internet'].cast(FloatType()))

#[7] Show correlation coefficient
print(november.stat.corr('measurement', 'SMS_total') )
print(november.stat.corr('measurement', 'Call_total') )
print(november.stat.corr('measurement', 'internet') )



