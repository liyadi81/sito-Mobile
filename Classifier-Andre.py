import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import MultilayerPerceptronClassifier, LogisticRegression, RandomForestClassifier

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

## Read in multiple parquets
data = spark.read.parquet('/mnt/sito/consumer_hh_data/part-00010-tid-3318985877776039692-74647555-e3f0-4ea3-9ce1-a0507c5c8fdc-0-c000.gz.parquet')
for i in [100, 130, 190, 350, 380]:
  b=('/mnt/sito/consumer_hh_data/part-00%i-tid-3318985877776039692-74647555-e3f0-4ea3-9ce1-a0507c5c8fdc-0-c000.gz.parquet' % (i))
  part=spark.read.parquet(b)
  data=data.unionAll(part)
  print(data.count())

## Parameter columns
paramCols = ['F14593', 'F3464',  'F3473',  'F9268', 'F5558', 'F7504', 'F10710', 'F10952', 'F3926']

## Setting dependent variables
X = data[paramCols]

#----------------------------------
## Do not run, testing cell to see how many null values 
X.where(X['F5558'].isNull()).count()
## Finding missingness per column
for i in X.columns:
  msng = X.where(X[i].isNull()).count()
  print i, ', ', msng
#----------------------------------

## Dropped missing values in column since not that many
X = X.filter(X.F5558.isNotNull())

## Imputed missing values with U for has a pet
X = X.fillna('U', subset='F9268')

#Need to drop this column since values don't make any sense...dictionary incorrect
X.select('F3471').distinct().show(33)
X = X.drop('F3471')

#----------------------------------
## Do not run, testing cell
for i in X.columns:
  print X.select(i).distinct().show()
## Checking how many classes in y(prediction column)
X.select('F5558').distinct().count()
## Describing columns for min/max, avg, etc
X.select('F9268', 'F5558', 'F7504', 'F10710', 'F10952').describe().show()
#----------------------------------

## Separating columns for String Indexer
idxCols = X.select([c for c in X.columns if c not in {'F5558'}])
regCols = [i for i in X.columns if i not in idxCols.columns]

## Checking how many classes in y (prediction column)
X.select('F14593').distinct().show()

#----------------------------------
## Do not run, testing cell
## Checking if there are any missing values left
for i in X.columns:
  msng = X.where(X[i].isNull()).count()
  print i, ', ', msng
#----------------------------------

## Setting columns that need to be indexed fo string indexer
initial = idxCols.columns
idxNames = [i+'_index' for i in initial]
idxNames = idxNames[3:7]

## Converting column to type double
X = X.withColumn("F3926", X["F3926"].cast('double'))

## Pipeline Functions:
indexers = [StringIndexer(inputCol=col, outputCol=col+'_index') for col in initial[3:7]] 

idxAssembler = VectorAssembler(inputCols=idxNames, 
                               outputCol="features")

rf = RandomForestClassifier(featuresCol="features", labelCol='F14593_index', numTrees=1000, minInstancesPerNode=4, featureSubsetStrategy='auto')

rfPipeline = Pipeline(stages=indexers+[idxAssembler, rf])

## Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = X.randomSplit([0.7, 0.3])

## Cache the training and test datasets.
trainingData.cache()
testData.cache()

## Fit/transforming 
rfModel = rfPipeline.fit(trainingData)
rfModel.write().overwrite().save('/mnt/need/need/rfModel')
rfResults = rfModel.transform(testData)

## Setting up confusion matrix for evaluation
display(rfResults.groupby('F14593_index', 'prediction').count().orderBy('F14593_index', 'prediction', ascending=True))
# Need to adjust plot options in Databricks





















