from google.colab import drive
drive.mount('/content/drive')

!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://archive.apache.org/dist/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz
!tar xf spark-3.1.1-bin-hadoop3.2.tgz
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.1.1-bin-hadoop3.2"

import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
spark

import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# WARNING, FILE "delay_clean_SVM.txt" is > 1.2 GB  --  added to gitignore

# Load and parse the data file, converting it to a DataFrame
clean = spark.read.format("libsvm").load('/content/drive/MyDrive/Colab_Notebooks/delay_clean_SVM.txt')
clean.show(5)
clean.printSchema()

# Number of rows in dataset
number_rows = clean.count()
number_rows
clean.groupBy('label').count().show()

# Index labels, adding metadata to the label column
# Fit on whole dataset to include all labels in index
labelIndexer = StringIndexer(inputCol = "label", outputCol = "indexedLabel").fit(clean)

# Automatically identify categorical features, and index them
# Set maxCategories so features with > 4 distinct values are treated as continuous
featureIndexer = VectorIndexer(inputCol = "features", outputCol = "indexedFeatures", maxCategories = 4).fit(clean)

from pyspark.ml.feature import Normalizer
normalizer = Normalizer(inputCol = "features", outputCol = "normFeatures", p = 1.0)
NormOutput = normalizer.transform(clean)

# Split the data into training and test sets
(trainingData, testData) = clean.randomSplit([0.7, 0.3])
trainingData.show(5)
testData.show(5)

# Oversampling performed to dataset
# https://medium.com/@junwan01/oversampling-and-undersampling-with-pyspark-5dbc25cdf253

from pyspark.sql.functions import col, explode, array, lit

major_df = clean.filter(col("label") == 0)
minor_df = clean.filter(col("label") == 1)
ratio = int(major_df.count()/minor_df.count())
print("Ratio of original dataset: {}".format(ratio)+" to 1 (on time : delayed flights)")
a = range(ratio)

# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')

# combine both oversampled minority rows and previous majority rows
combined_df = major_df.unionAll(oversampled_df)
combined_df.show()
combined_df.groupBy('label').count().show()

# Split the data into training and test sets
(trainingData, testData) = combined_df.randomSplit([0.7, 0.3])
trainingData.show(5)
testData.show()

# Gradient-boosted tree classifier (GBT)
# Train a GBT model
gbt = GBTClassifier(labelCol = "indexedLabel", featuresCol = "indexedFeatures", maxIter = 30, maxDepth = 10,
                    stepSize = 1)

# Chain indexers and GBT in a Pipeline
pipeline = Pipeline(stages = [labelIndexer, featureIndexer, gbt])

# Train model.  This also runs the indexers
import time
start_time = time.time()
model = pipeline.fit(trainingData)
print("Training Time: %s seconds" % (str(time.time() - start_time)))

# Make predictions
predictions = model.transform(testData)

# Select example rows to display
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol = "indexedLabel", predictionCol = "prediction", metricName = "accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)

y_true = predictions.select(['indexedLabel']).collect()
y_pred = predictions.select(['prediction']).collect()

print(confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred))

importanceSummary = gbtModel.featureImportances
importanceSummary

from matplotlib import pyplot as plt
plt. bar(x = range (len (importanceSummary)), height = importanceSummary)
plt.show()

# Saving trained model
#model_path = "./drive/" + "./MyDrive/" + "./SavedModels/" + "GBTmodel"
#model.write().overwrite().save(model_path)

# Random forest classifier (RFC)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString

# Train a RandomForest model
rf = RandomForestClassifier(labelCol = "indexedLabel", featuresCol = "indexedFeatures", numTrees = 40)

# Convert indexed labels back to original labels
labelConverter = IndexToString(inputCol = "prediction", outputCol = "predictedLabel",
                               labels = labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages = [labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers
import time
start_time = time.time()
model = pipeline.fit(trainingData)
print("Training Time: %s seconds" % (str(time.time() - start_time)))

# Make predictions
predictions1 = model.transform(testData)

# Select example rows to display
predictions1.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol = "indexedLabel", predictionCol = "prediction", metricName = "accuracy")
accuracy = evaluator.evaluate(predictions1)
print("Accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)
print(rfModel.featureImportances)

from matplotlib import pyplot as plt
plt. bar(x = range (len (rfModel.featureImportances)), height = rfModel.featureImportances)
plt.show()

y_true = predictions1.select(['indexedLabel']).collect()
y_pred = predictions1.select(['prediction']).collect()

print(confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred))

# Saving trained model
#model_path = "./drive/" + "./MyDrive/" + "./SavedModels/" + "RFmodel"
#model.write().overwrite().save(model_path)

# Factorization machines classifier
from pyspark.ml.classification import FMClassifier
from pyspark.ml.feature import MinMaxScaler

# Index labels, adding metadata to the label column
# Fit on whole dataset to include all labels in index
labelIndexer = StringIndexer(inputCol = "label", outputCol = "indexedLabel").fit(clean)

# Scale features
featureScaler = MinMaxScaler(inputCol = "features", outputCol = "scaledFeatures").fit(clean)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = clean.randomSplit([0.7, 0.3])

# Train a FM model
fm = FMClassifier(labelCol = "indexedLabel", featuresCol = "scaledFeatures", stepSize = 0.01)

# Create a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureScaler, fm])

# Train model
start_time = time.time()
model = pipeline.fit(trainingData)
print("Training Time: %s seconds" % (str(time.time() - start_time)))

# Select (prediction, true label) and compute test accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol = "indexedLabel", predictionCol = "prediction", metricName = "accuracy")
accuracy = evaluator.evaluate(predictions2)
print("Test set accuracy = %g" % accuracy)
print("Test Error = %g" % (1.0 - accuracy))

y_true = predictions2.select(['indexedLabel']).collect()
y_pred = predictions2.select(['prediction']).collect()

print(confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred))

#model_path = "./drive/" + "./MyDrive/" + "./SavedModels/" + "FMmodel"
#model.write().overwrite().save(model_path)