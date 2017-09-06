import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import KMeans

import matplotlib.pyplot as plt
%matplotlib inline

## Mount S3 bucket nycdsabootcamp to the Databricks File System
s3Path = "s3a://{0}:{1}@{2}".format("XXX", 
                                    "XXX", 
                                    "nycdsabootcamp/")
mntPath = "/mnt/data/"
try:
  dbutils.fs.mount(s3Path, mntPath)
except:
  pass

## read in parquet
data = spark.read.parquet('/mnt/sito/consumer_hh_data/part-00490-tid-3318985877776039692-74647555-e3f0-4ea3-9ce1-a0507c5c8fdc-0-c000.gz.parquet')

## double checking compiled data frame
# print data.count()
# print len(data.columns)

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

# print len(catList)
# print len(numList)

## labeling flat list of categorical columns
cat=[]
for i in range(len(catList)):
  cat_x = catList.items()[i][1]
  cat += [[cat_x]]
catCols = [item for sublist in cat for item in sublist]

## labeling flat list of numerical columns
num=[]
for i in range(len(numList)):
  num_x = numList.items()[i][1]
  num += [[num_x]]
numCols = [item for sublist in num for item in sublist]

## Creating dataframes for numerical and categorical features
catData = data[catCols]
numData = data[numCols]

## Converting numerical vaues to doubles
for i in numData.columns:
  numData = numData.withColumn(i, numData[i].cast('double'))

## Imputing missing values with 0 
numData = numData.fillna(0)

## Making dense feature vector
assembler = VectorAssembler(inputCols=numCols, 
                            outputCol="features")
output = assembler.transform(numData)

## scaling data by using std dev
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True)
# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(output)
# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(output)

## Run PCA to reduce dimensions
pcaExtracted = PCA(k=200, inputCol='features', outputCol='pcaFeatures')
pcaModel = pcaExtracted.fit(scaledData)
pcaResult = pcaModel.transform(scaledData).select('pcaFeatures')

## Re-scale data for K-means
scaler = StandardScaler(inputCol="pcaFeatures", outputCol="features", withStd=True)
# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(pcaResult)
# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(pcaResult)

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
#     ax.title('Elbow method')
#     ax.xlabel('Number of clusters')
#     ax.ylabel('Inertia')
#     ax.show()
    display(fig)


























