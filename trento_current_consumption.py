from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import pyspark.sql.functions as func
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from numpy import polyfit


plt.style.use('seaborn')

spark = SparkSession\
    .builder\
    .enableHiveSupport()\
    .getOrCreate()\

#[1] Loading the electricity dataset and creating a dataframe
lines = spark.read.option("delimiter", ",")\
    .csv('inputs/electricity/line.csv')\
    .toDF('square_id', 'line_id', 'ubication_number')

consumptions = spark.read.option("delimiter", ",")\
    .csv('inputs/electricity/SET-nov-2013.csv')\
    .toDF('line_id', 'time_istant', 'consumption')

electricity = lines.join(consumptions, on='line_id', how='left')


#[2] For simplification I will resample so that each row represent a whole hour
electricity = electricity.withColumn("hour", func.substring("time_istant", 0, 13))
electricity = electricity.groupBy(electricity.hour,electricity.square_id,)\
    .agg(func.avg("consumption").alias("total_consumption"))

electricity = electricity.withColumn('total_consumption', electricity['total_consumption'].cast(FloatType()))
electricity = electricity.withColumn('ubication_number', electricity['ubication_number'].cast(FloatType()))
electricity = electricity.dropna()



#[3] Clustering
vecAssembler = VectorAssembler(inputCols=["total_consumption", "ubication_number"], outputCol="features")
new_df = vecAssembler.transform(electricity).select('square_id', 'hour', 'line_id', 'features')
new_df.show()
cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(new_df.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(new_df)
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')

k = 15
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(new_df)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)


#[4] Import precipitation dataset
precipitation = spark.read.option("delimiter", ",")\
    .csv('inputs/electricity/precipitation-trentino.csv')\
    .toDF('time_istant', 'square_id', 'rain_intensity')
precipitation = precipitation.withColumn('date_again', func.from_unixtime(func.unix_timestamp(precipitation.time_istant, 'yyyyMMddHHmm'),'yyyy-MM-dd HH:mm'))
precipitation = precipitation.withColumn("hour", func.substring("date_again", 0, 13))
precipitation = precipitation.groupBy(precipitation.hour,precipitation.square_id)\
    .agg(func.avg("rain_intensity").alias("rain"))
precipitation = precipitation.withColumn('rain', func.round("rain"))

joinTable = electricity.join(precipitation, ['hour','square_id'])
joinTable.show()
joinTable = joinTable.withColumn('total_consumption', joinTable['total_consumption'].cast(FloatType()))
joinTable = joinTable.withColumn('rain', joinTable['rain'].cast(FloatType()))
joinTable = joinTable.dropna()
joinTable.show(100)

#[5] Linear Regression
x1 = joinTable.toPandas()['rain'].values.tolist()
y1 = joinTable.toPandas()['total_consumption'].values.tolist()
plt.scatter(x1, y1, color='red', s=30)
plt.xlabel('Piogge')
plt.ylabel('Consumi')
plt.title('Linear Regression')
p1 = polyfit(x1, y1, 1)
plt.plot(x1, np.polyval(p1,x1), 'g-' )
plt.show()


joinTable2 = joinTable.select(joinTable.rain,joinTable.square_id, joinTable.total_consumption.alias('label'))
train, test = joinTable2.randomSplit([0.7, 0.3])
assembler = VectorAssembler().setInputCols(['rain',])\
.setOutputCol('features')
train01 = assembler.transform(train)
train02 = train01.select("features","label")
train02.show(truncate=False)

lr = LinearRegression()
model = lr.fit(train02)
test01 = assembler.transform(test)
test02 = test01.select('features', 'label')
test03 = model.transform(test02)
test03.show(truncate=False)
evaluator = RegressionEvaluator()
print(evaluator.evaluate(test03,{evaluator.metricName: "r2"}))
print(evaluator.evaluate(test03,{evaluator.metricName: "mse"}))
print(evaluator.evaluate(test03,{evaluator.metricName: "rmse"}))
print(evaluator.evaluate(test03,{evaluator.metricName: "mae"}))

