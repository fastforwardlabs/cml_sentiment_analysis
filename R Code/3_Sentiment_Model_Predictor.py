## R Code Part 3: Model Serving
#
# This script explains how to create and deploy Models in CML which function as a 
# REST API to serve predictions. This feature makes it very easy for a data scientist 
# to make trained models available and usable to other developers and data scientists 
# in your organization.
#
# If you haven't yet, run through the initialization steps in the README file and Part 1 and Part 2. 
# The model file is fetched from the object storage set up in the previous steps.
#
### Requirements
# Models have the following requirements:
# - model code in a `.R` script, not a notebook
# - a `cdsw-build.sh` script containing code to install all dependencies
#
# > In addition, Models *must* be designed with one main function that takes a dictionary as its sole argument
# > and returns a single dictionary.
# > CML handles the JSON serialization and deserialization.
#
# In this file, there is minimal code since calculating predictions is much simpler 
# than training a machine learning model.
#
# When a Model API is called, CML will translate the input and returned JSON blobs to and from R lists.
# Thus, the script simply loads the model we saved at the end of the last section,
# passes the input dictionary into the model, and returns the results as a list with the following format:
# ```   
#    {
#       "sentiment" : sentiment, 
#       "confidence" : conf 
#    }
#```
# The Model API will return this list serialized as JSON.
# 

### Creating and deploying a Model
# To deploy the model trailed in the previous step, from the Project page, click **Models > New
# Model** and create a new model with the following details:
#
# * **Name**: r_model
# * **Description**: R model
# * **Enable Authentication**: [ ] _unchecked_
# * **File**: R Code/3_Sentiment_Model_Predictor.py
# * **Function**: predict_sentiment
# * **Input**: 
# ```
# {
# 	"sentence":"I'm no dunce, I was born an oaf and I'll die an oaf"
# }
# ```
# * **Kernel**: R
# * **Engine Profile**: 1vCPU / 4 GiB Memory

# Leave the rest unchanged. Click **Deploy Model** and the model will go through the build 
# process using `cdsw-build.sh` and deploy a REST endpoint. Once the model is deployed, you 
# can test it's working from the model Model Overview page. 
# After accepting the dialog, CML will *build* a new Docker image ,
# then *assign an endpoint* for sending requests to the new Model.

## Testing the Model
# > To verify it's returning the right results in the format you expect, you can 
# > test any Model from it's *Overview* page.
#
# If you entered an *Example Input* before, it will be the default input here, 
# though you can enter your own.

## Using the Model
#
# > The *Overview* page also provides sample `curl` or Python commands for calling your Model API.
# > You can adapt these samples for other code that will call this API.
#
# This is also where you can find the full endpoint to share with other developers 
# and data scientists.
#
# **Note:** for security, you can specify 
# [Model API Keys](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-model-api-key-for-models.html) 
# to add authentication.

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Word2VecModel, Tokenizer, StopWordsRemover
from pyspark.sql.functions import regexp_replace
import os

spark = SparkSession.builder \
      .appName("Sentiment") \
      .master("local[*]") \
      .config("spark.driver.memory","4g")\
      .config("spark.hadoop.yarn.resourcemanager.principal",os.getenv("HADOOP_USER_NAME"))\
      .getOrCreate()

storage = os.getenv("STORAGE")      

tokenizer = Tokenizer(inputCol="spoken_words", outputCol="word_list")
remover = StopWordsRemover(inputCol="word_list", outputCol="wo_stop_words")
w2v_model_fitted = Word2VecModel.load(storage + "/datalake/data/sentiment/w2v_model_fitted")
lr_model = PipelineModel.load(storage + "/datalake/data/sentiment/lr_model")

#args = {"sentence":"I'm no dunce, I was born an oaf and I'll die an oaf"}

def predict_sentiment(args):
  input_sentence = args["sentence"]#.split(",")
  sentence_df = spark.createDataFrame([(input_sentence,)],['spoken_words'])
  sentence_df = sentence_df.select(regexp_replace('spoken_words',r'[_\"\'():;,.!?\\-]', ' ').alias('spoken_words'))
  sentence_df = tokenizer.transform(sentence_df)
  sentence_df = remover.transform(sentence_df)
  sentence_df = w2v_model_fitted.transform(sentence_df)
  result = lr_model.transform(sentence_df).collect()[0]
  if result.prediction==0:
    sentiment = 'Negative'
    conf = round(result.probability[0] * 100,3)
  else:
    sentiment = 'Positive'  
    conf = round(result.probability[1] * 100,3)
  return {"sentiment" : sentiment, "confidence" : conf }

