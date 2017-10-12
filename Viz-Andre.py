import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

kmeans = KMeans()
import matplotlib.pyplot as plt

## Mount S3 bucket nycdsabootcamp to the Databricks File System
s3Path = "s3a://{0}:{1}@{2}".format("AKIAI2P5MSEO2JYXJVQQ", 
                                    "YJboxXSbraX4rg17aqtI+HmBjWCcpu4dxv2HW+bm", 
                                    "nycdsabootcamp/")
mntPath = "/mnt/data/"
try:
  dbutils.fs.mount(s3Path, mntPath)
except:
  pass

data = spark.read.parquet('/mnt/sito/consumer_hh_data/part-00010-tid-3318985877776039692-74647555-e3f0-4ea3-9ce1-a0507c5c8fdc-0-c000.gz.parquet')
for i in [100, 130, 190, 350, 380]:
  b=('/mnt/sito/consumer_hh_data/part-00%i-tid-3318985877776039692-74647555-e3f0-4ea3-9ce1-a0507c5c8fdc-0-c000.gz.parquet' % (i))
  part=spark.read.parquet(b)
  data=data.unionAll(part)
  print(data.count())

## Bar chart
def pltBar(xVal, yVal, data, title):
  vizData = data.select(xVal, yVal)
  grpViz = vizData\
            .groupBy(xVal)\
            .agg(mean(vizData[yVal]).alias("Avg_{0}".format(yVal)), \
                  stddev(vizData[yVal]).alias("Std_Dev_{0}".format(yVal)),\
                  count(vizData[xVal]).alias('Count'),\
                  min(vizData[yVal]).alias('Min_{0}_value'.format(yVal)),\
                  max(vizData[yVal]).alias('Max_{0}_value'.format(yVal)))\
            .orderBy(xVal, ascending=True)
  grpViz = grpViz.toPandas()
  plt.clf()
  fig = plt.figure(1, figsize=(9, 6))
  ax = sns.barplot(x=xVal, y=("Avg_{0}".format(yVal)), data=grpViz, color='blue', alpha=.2)
  plt.tick_params(labelsize=4)
  plt.title(title, fontsize = 24)
  plt.xlabel(xVal, fontsize = 14)
  plt.ylabel(yVal, fontsize = 14)
  display(fig)

pltBar(xVal='F7504', yVal='F3464', data=data, title='Median Home Value by Population Density')
pltBar(xVal='F3753', yVal='F14592', data=data, title='Income by Age')
pltBar(xVal='F3753', yVal='F3931', data=data, title='Women\'s Mid-Range Apparel Purchases by Age')
pltBar(xVal='F3753', yVal='F3927', data=data, title='Women\'s High-end Apparel Purchases by Age')
pltBar(xVal='F3753', yVal='F7035', data=data, title='Electronic Gadgets by Age')

def pltBar2(xVal, data, title):
  vizData = data.select(xVal)
  grpViz = vizData\
            .groupBy(xVal)\
            .agg(count(vizData[xVal]).alias('Count'))\
            .orderBy(xVal, ascending=True)
  grpViz = grpViz.toPandas()
  plt.clf()
  fig = plt.figure(1, figsize=(9, 6))
  ax = sns.barplot(x=xVal, y='Count', data=grpViz, color='blue', alpha=.2)
  plt.tick_params(labelsize=4)
  plt.title(title, fontsize = 24)
  plt.xlabel(xVal, fontsize = 14)
  plt.ylabel(yVal, fontsize = 14)
  display(fig)

pltBar2(xVal='F3753', data=data, title='Age Distribution')

## Purchase Power Bar charts
def pltBarPurchPower(xVal, yVal, data, title):
  vizData = data.select(xVal, yVal)
  grpViz = vizData\
            .groupBy(xVal)\
            .agg(mean(vizData[yVal]).alias("Avg_{0}".format(yVal)), \
                  stddev(vizData[yVal]).alias("Std_Dev_{0}".format(yVal)),\
                  count(vizData[xVal]).alias('Count'),\
                  min(vizData[yVal]).alias('Min_{0}_value'.format(yVal)),\
                  max(vizData[yVal]).alias('Max_{0}_value'.format(yVal)))
  grpViz = grpViz\
            .withColumn('Purch_Power', grpViz["Avg_{0}".format(yVal)]*grpViz['Count'])\
            .orderBy(xVal, ascending=True)
  grpViz = grpViz.toPandas()
  plt.clf()
  fig = plt.figure(1, figsize=(9, 6))
  ax = sns.barplot(x=xVal, y='Purch_Power', data=grpViz, color='blue', alpha=.2)
  plt.tick_params(labelsize=4)
  plt.title(title, fontsize = 24)
  plt.xlabel(xVal, fontsize = 14)
  plt.ylabel('Purchasing Power', fontsize = 14)
  display(fig)

pltBarPurchPower(xVal='F3753', yVal='F7035', data=data, title='Electronic Gadgets by Age')
pltBarPurchPower(xVal='F3753', yVal='F3931', data=data, title='Women\'s Mid-Range Apparel Purchases by Age')
pltBarPurchPower(xVal='F3753', yVal='F3927', data=data, title='Women\'s High-end Apparel Purchases by Age')




























