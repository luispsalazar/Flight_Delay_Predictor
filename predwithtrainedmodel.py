#################################################################
# New untrained dataset must be named: "new_data_clean_SVM.txt" #
#################################################################

from google.colab import drive
drive.mount('/content/drive')
get_ipython().system('ls')

get_ipython().system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')
get_ipython().system('wget -q http://archive.apache.org/dist/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz')
get_ipython().system('tar xf spark-3.1.1-bin-hadoop3.2.tgz')
get_ipython().system('pip install -q findspark')

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.1.1-bin-hadoop3.2"

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import PipelineModel

# # Load trained model
# Model's path
model_path = "./drive/" + "./MyDrive/" + "./SavedModels/" + "GBTmodel"

# Load trained model
loadedModel = PipelineModel.load(model_path)

# Load and parse the data file, converting it to a DataFrame
clean = spark.read.format("libsvm").load('/content/drive/MyDrive/Colab_Notebooks/new_data_clean_SVM.txt')

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

# Make predictions with loaded model
predictions = loadedModel.transform(clean)

# Select example rows to display from prediction
predictions.select("prediction", "indexedLabel", "features").show()