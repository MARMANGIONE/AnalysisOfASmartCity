# -*- coding: utf-8  -*-
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# ----------------------------------------------------------------------------------------------------------------------
#Analysis with Spark
#[1] Data ingestion for all november month
spark = SparkSession\
    .builder\
    .enableHiveSupport()\
    .getOrCreate()\


sparkDf = spark.read.option("delimiter", "\t")\
    .csv('inputs/telecom/november/*.txt')\
    .toDF('square_id', 'time_interval', 'country_code', 'SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'internet_traffic')\


sparkDf.printSchema()
sparkDf = sparkDf.withColumn('date_again', from_unixtime(sparkDf.time_interval / 1000, 'yyyy-MM-dd HH:mm:ss:SSS"'))
sparkDf = sparkDf.withColumn('date_again', unix_timestamp(sparkDf.date_again, 'yyyy-MM-dd HH:mm:ss:SSS"').cast('timestamp'))
sparkDf = sparkDf.withColumn('SMS_in', sparkDf['SMS_in'].cast(FloatType()))
sparkDf = sparkDf.withColumn('SMS_out', sparkDf['SMS_out'].cast(FloatType()))
sparkDf = sparkDf.withColumn('weekday', dayofweek(sparkDf['date_again']))


sparkDf.createOrReplaceTempView("telecommunicationData")


smsDF = sparkDf.groupBy(sparkDf.weekday)\
    .agg(sum("SMS_out").alias("smsInUscita"),sum("SMS_in").alias("smsInEntrata"))
smsDF = smsDF.withColumn("total", smsDF["smsInEntrata"] + smsDF["smsInUscita"]).show()

callDF = sparkDf.groupBy(sparkDf.weekday)\
    .agg(sum("Call_out").alias("chiamateInUscita"),sum("Call_in").alias("chiamateInEntrata"))
callDF = callDF.withColumn("TotaleChiamate", callDF["chiamateInUscita"] + callDF["chiamateInEntrata"]).show()

sparkDf.groupBy(sparkDf.weekday)\
    .agg(max("internet_traffic"))\
    .show()
# ----------------------------------------------------------------------------------------------------------------------


# Graphic Analysis with Pandas

#[1] Loading the dataset and creating a dataframe
sns.set_style("ticks")
sns.set_context("paper")

NOVEMBER_PATH = 'inputs/telecom/november/sms-call-internet-mi-2013-11-'
DECEMBER_PATH = 'inputs/telecom/december/sms-call-internet-mi-2013-12-'

november_interval = range(04,11)
december_interval = range(23,30)

parse = lambda x: datetime.datetime.fromtimestamp(float(x)/1000)

dataframe = pd.DataFrame({})
for index in november_interval:
    df = pd.read_csv(NOVEMBER_PATH + str(index).zfill(2) +'.txt', sep='\t', encoding="utf-8-sig", names=['square_id', 'time_interval', 'country_code', 'SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'internet_traffic'], parse_dates=['time_interval'], date_parser=parse)
    df = df.set_index('time_interval')
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df = df.groupby(['hour', 'weekday', 'square_id'], as_index=False).sum()

    dataframe = dataframe.append(df)

dataframe['idx'] = dataframe['hour'] + (dataframe['weekday']*24)

#[2] Group by weekday-hour
dataframe_milano = dataframe.groupby(['weekday', 'hour'], as_index=False).sum()
dataframe_milano['sms'] = dataframe_milano['SMS_in'] + dataframe_milano['SMS_out']
dataframe_milano['Chiamate'] = dataframe_milano['Call_in'] + dataframe_milano['Call_out']
dataframe_milano.rename(columns={'sms': 'SMS', 'internet_traffic': 'Internet'}, inplace=True)

#[3] Behaviour plot
types = ['SMS', 'Chiamate', 'Internet']
f, axs = plt.subplots(len(types), sharex=True, sharey=True)


#[4] Z-score
sliceSum_z = (dataframe_milano - dataframe_milano.mean()) / dataframe_milano.std()
print(sliceSum_z)

#[5] Events-Plot
for i, p in enumerate(types):
    plt.xticks(np.arange(168, step=10))
    axs[i].plot(sliceSum_z[p], label=p)
    axs[i].legend(loc='upper center')
    sns.despine()

f.text(0, 0.5, "Numero di eventi", rotation="vertical", va="center")

plt.xlabel("Ore in una settimana")
plt.savefig('eventi.png', format='png', dpi=330, bbox_inches='tight')


f, axs = plt.subplots(1, sharex=True, sharey=False)

axs.plot(dataframe[dataframe.square_id == 5060].set_index('idx')['internet_traffic'], label='Duomo')
axs.plot(dataframe[dataframe.square_id == 5737].set_index('idx')['internet_traffic'], label='San Siro')
axs.plot(dataframe[dataframe.square_id == 4456].set_index('idx')['internet_traffic'], label='Navigli')
axs.plot(dataframe[dataframe.square_id == 5356].set_index('idx')['internet_traffic'], label='Parco Sempione')

axs.set_xticklabels([])
sns.despine()


box = axs.get_position()
axs.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=5)
sns.despine()

f.text(0, 0.5, u"Utilizzo di internet in luoghi di interesse", rotation="vertical", va="center")
plt.savefig('luoghidiinteresse.png', format='png', dpi=330, bbox_inches='tight')

#[6] Boxplots
boxplots = {
    'Chiamate': "Chiamate",
    'sms': "SMS",
    "internet_traffic": "Internet"
}
f, axs = plt.subplots(len(boxplots.keys()), sharex=True, sharey=False)
f.subplots_adjust(hspace=.35,wspace=0.1)
dataframe['sms'] = dataframe['SMS_in'] + dataframe['SMS_out']
dataframe['Chiamate'] = dataframe['Call_in'] + dataframe['Call_out']
i = 0
plt.suptitle("")
for k, v in boxplots.iteritems():

    ax = dataframe.reset_index().boxplot(column=k, by='weekday', grid=False, sym='', ax =axs[i])
    axs[i].set_title(v)
    axs[i].set_xlabel("")
    sns.despine()
    i += 1

plt.xlabel(u"Giorni della settimana (0=Lunedì, 6=Domenica)")
f.text(0, 0.5, u"Numero di eventi", rotation="vertical", va="center")
plt.savefig('boxplots-Milano.png', format='png', dpi=330,bbox_inches='tight')

