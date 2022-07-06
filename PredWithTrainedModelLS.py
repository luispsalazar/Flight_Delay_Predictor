#################################################################
# New untrained dataset must be named: "new_data_clean_SVM.txt" #
#################################################################

import os
# Find the latest version of spark 3.0  from http://www.apache.org/dist/spark/ and enter as the spark version
# For example:
# spark_version = 'spark-3.0.3'
spark_version = 'spark-3.3.0'
os.environ['SPARK_VERSION']=spark_version

# Install Spark and Java
# !apt-get update
# !apt-get install openjdk-11-jdk-headless -qq > /dev/null
# !wget -q http://www.apache.org/dist/spark/$SPARK_VERSION/$SPARK_VERSION-bin-hadoop2.7.tgz
# !tar xf $SPARK_VERSION-bin-hadoop2.7.tgz
# !pip install -q findspark

# Set Environment Variables
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["JAVA_HOME"] = "/usr/local/Cellar/openjdk/18.0.1.1/libexec/openjdk.jdk"
os.environ["SPARK_HOME"] = f"/Users/luispsalazar/Documents/Dev/Apache-Spark/spark-3.3.0-bin-hadoop3"

# Start a SparkSession
import findspark
findspark.init()

# Start Spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DataFrameBasics").getOrCreate()

from pyspark.ml import Pipeline
# from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
from pyspark.ml.feature import StringIndexer, VectorIndexer
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import PipelineModel

# # Load trained model
# Model's path
#model_path = "./drive/" + "./MyDrive/" + "./SavedModels/" + "GBTmodel"
model_path = "SavedModels/" + "GBTmodel"

# Load trained model
#loadedModel = PipelineModel.load(model_path)
loadedModel = PipelineModel.load(model_path)
print(loadedModel)

# Load and parse the data file, converting it to a DataFrame
#clean = spark.read.format("libsvm").load('/content/drive/MyDrive/Colab_Notebooks/new_data_clean_SVM.txt')
clean = spark.read.format("libsvm").load('Resources/new_data_clean_SVM.txt')

# Number of rows in dataset
number_rows = clean.count()
print(number_rows)

# clean.groupBy('label').count().show()

# # Index labels, adding metadata to the label column
# # Fit on whole dataset to include all labels in index
# labelIndexer = StringIndexer(inputCol = "label", outputCol = "indexedLabel").fit(clean)

# # Automatically identify categorical features, and index them
# # Set maxCategories so features with > 4 distinct values are treated as continuous
# featureIndexer = VectorIndexer(inputCol = "features", outputCol = "indexedFeatures", maxCategories = 4).fit(clean)

# from pyspark.ml.feature import Normalizer

# normalizer = Normalizer(inputCol = "features", outputCol = "normFeatures", p = 1.0)
# NormOutput = normalizer.transform(clean)

# # Make predictions with loaded model
# predictions = loadedModel.transform(clean)

# # Select example rows to display from prediction
# predictions.select("prediction", "indexedLabel", "features")
# print(predictions)
#sc.stop()