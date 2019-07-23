from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

spark = SparkSession\
    .builder\
    .enableHiveSupport()\
    .getOrCreate()\

#[1] Loading the dataset and creating a dataframe
df = spark.read.option('timestampFormat', 'MM/d/yyyy HH:mm')\
    .option("delimiter", "\t")\
    .csv('inputs/telecom/november/*.txt')\
    .toDF('square_id', 'time_interval', 'country_code', 'SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet_traffic')

df.printSchema()

cleanedData = df.filter(df['country_code'] > 0)
cleanedData.count()

for column in df.columns:
    if column!='time_interval':
        df.describe(column).show

#[2] Analysing the square_id who has most calls, sms, and internet usage.
cleanedData.createOrReplaceTempView("telecommunicationData")
spark.sql("select square_id, SMS_out from telecommunicationData where SMS_out in (select max(SMS_out) from telecommunicationData)").show()
spark.sql("select square_id, Call_out from telecommunicationData where Call_out in (select max(Call_out) from telecommunicationData)").show()
spark.sql("select square_id, Internet_traffic from telecommunicationData where Internet_traffic in (select max(Internet_traffic) from telecommunicationData)").show()

# Calculates the sum of activity of the different uses(GSM,SMS,Internet) for a group of countries
aggcountryDF = spark.sql("""select 
                                CASE country_code
                                    WHEN 63 THEN "Filippine"
                                    WHEN 33 THEN "Francia"
                                    WHEN 34 THEN "Spagna"
                                    WHEN 39 THEN "Italia"
                                    WHEN 44 THEN "Inghilterra"
                                    WHEN 20 THEN "Egitto"
                                    WHEN 48 THEN "Polonia"
                                    WHEN 49 THEN "Germania"
                                    WHEN 351 THEN "Portogallo"
                                    WHEN 86 THEN "Cina"
                                    ELSE "Altre"
                                END as country,
                                round(sum(SMS_in),6) SMS_in, 
                                round(sum(SMS_out),6) SMS_out, 
                                round(sum(Call_in),6) Call_in, 
                                round(sum(Call_out),6) Call_out, 
                                round(sum(Internet_traffic),6) Internet_traffic
                            from telecommunicationData
                            where country_code != 0
                            group by country
                            order by 1
                            """)
aggcountryDF.show()

#[3] Show the results
plotdf = aggcountryDF.where("country not in ('Italia','Altre')").toPandas()
print(plotdf)
plotdf.plot(kind='barh', x='country', y=['SMS_in', 'SMS_out'], colormap='tab20b')
plt.show()
plotdf.plot(kind='barh', x='country', y=['Call_in', 'Call_out'], colormap='tab10')
plt.show()
plotdf.plot(kind='barh', x='country', y=['Internet_traffic'], colormap='summer')
plt.show()

spark.stop()
