import pandas as pd
import numpy as np
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.mllib.linalg import Vectors, VectorUDT

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

%fs
ls /mnt/sito/consumer_hh_data/

data = spark.read.parquet('/mnt/sito/consumer_hh_data/part-00490-tid-3318985877776039692-74647555-e3f0-4ea3-9ce1-a0507c5c8fdc-0-c000.gz.parquet')
data = data.drop('loc')

## script to return all numeric columns
## can use this to select columns with numeric values for PCA
numList=dict()
catList=dict()
for index, i in enumerate(data.columns):
  avg_i=data.select(avg(i))
  if (avg_i.filter('avg(%s) = 0' %(i)).count()==1):
    catList[index]=i
  elif (avg_i.filter(('avg(%s) is null') %(i)).count()==1):
    catList[index]=i
  else:
    numList[index]=i

## Separating categorical columns from numerical
cat=[]
for i in range(len(catList)):
  cat_x = catList.items()[i][1]
  cat += [[cat_x]]
catCols = [item for sublist in cat for item in sublist]

num=[]
for i in range(len(numList)):
  num_x = numList.items()[i][1]
  num += [[num_x]]
numCols = [item for sublist in num for item in sublist]

catData = data[catCols]
numData = data[numCols]

## Converting numerical vaues to doubles
for i in numData.columns:
  numData = numData.withColumn(i, numData[i].cast('double'))

## Imputing missing values with mean
imputer = Imputer(inputCols=numCols, outputCols=numCols).setStrategy("mean")
numData_imp = imputer.fit(numData).transform(numData)

## Making dense feature vector
assembler = VectorAssembler(inputCols=numCols, 
                            outputCol="features")

output = assembler.transform(numData_imp)

## scaling data by using std dev
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True)
scalerModel = scaler.fit(output)
scaledData = scalerModel.transform(output)

## Run PCA to reduce dimensions
pcaExtracted = PCA(k=200, inputCol='features', outputCol='pcaFeatures')
pcaModel = pcaExtracted.fit(output)
pcaResult = pcaModel.transform(output).select('pcaFeatures')

## Function to find optimal # of clusters
def plot_inertia(km, X, n_cluster_range):
    inertias = []
    for i in n_cluster_range:
        km = KMeans().setK(i)
        model = km.fit(X)
        wssse = model.computeCost(X)
        inertias.append(wssse)
    fig, ax = plt.subplots()
    ax.plot(n_cluster_range, inertias, marker='o')
    display(fig)

plot_inertia(kmeans, scaledData2, [50, 100, 200, 350, 500, 1000])

## Re-scale data for K-means
scaler = StandardScaler(inputCol="pcaFeatures", outputCol="features", withStd=True)
scalerModel = scaler.fit(pcaResult)
scaledData2 = scalerModel.transform(pcaResult)

## Run K-means model
km = kmeans.setK(350)
model = km.fit(scaledData2)
model.save('/mnt/need/need/modelParq490_k350')
modelTransf = model.transform(scaledData2)






























