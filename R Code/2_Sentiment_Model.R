## R Code Part 2: Model Training
#
# This script is used to train an Sentiment Analysis model and also how to use the
# Jobs to run model training and the Experiments feature of CML to facilitate model
# tuning.
#
# If you haven't yet, run through the initialization steps in the README file and Part 1.
# In Part 1, the data is imported into the `default.simpsons_spark_table`
# and the `default.afinn_table` table in Hive. All data accesses in the file fetch from Hive.
#
# To simply train the model once, run this file in an R workbench session.
#
# There other ways of running the model training process is using a job.
#
# ***Scheduled Jobs***
#
# The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html)**
# feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model 
# training process as a job, create a new job by going to the Project window and clicking **Jobs >
# New Job** and entering the following settings:
# * **Name** : Train R Model
# * **Script** : R Code/2_Sentiment_Model.R
# * **Arguments** : _Leave blank_
# * **Kernel** : R
# * **Schedule** : Manual
# * **Engine Profile** : 1 vCPU / 4 GiB
# The rest can be left as is. Once the job has been created, click **Run** to start a manual 
# run for that job.

### Load in the Data
# The following steps load up the 2 tables saved in step 1 into a spark dataframe using Sparklyr.

library(sparklyr)
library(dplyr)

storage <- Sys.getenv("STORAGE")

config <- spark_config()
config$spark.executor.memory <- "4g"
config$spark.executor.instances <- "3"
config$spark.executor.cores <- "4"
config$spark.driver.memory <- "2g"
config$spark.yarn.access.hadoopFileSystems <- storage
sc <- spark_connect(master = "yarn-client", config=config)

spark_read_table(sc,"simpsons_spark_table")
spark_read_table(sc,"afinn_table")

simpsons_spark_table <- tbl(sc, "simpsons_spark_table")
afinn_table <- tbl(sc, "afinn_table")

## Create Sentiment values per Sentence
# This is explained in the previous step and its how to create a sentiment value for each 
# line of dialogue. 

sentences <- simpsons_spark_table %>%  
  mutate(word = explode(wo_stop_words)) %>% 
  select(spoken_words, word) %>%  
  filter(nchar(word) > 2) %>% 
  compute("simpsons_spark_table")

sentence_values <- sentences %>% 
  inner_join(afinn_table) %>% 
  group_by(spoken_words) %>% 
  summarise(weighted_sum = sum(value))

### Convert value to Binary Label
# As there is a range of values, the next step take the mean value as the center point for Postive vs 
# Negative sentiment and adds the `sent_score` column which will be used as the dependant variable
# to train the model.

weighted_sum_summary <- sentence_values %>% sdf_describe(cols="weighted_sum")

weighted_sum_mean <- as.data.frame(weighted_sum_summary)$weighted_sum[2]

sentence_scores <- sentence_values %>% 
  mutate(sent_score = ifelse(weighted_sum > weighted_sum_mean,1,0))

sentence_values_tokenized <- 
  sentence_scores %>% 
  ft_tokenizer(input_col="spoken_words",output_col= "word_list") %>%
  ft_stop_words_remover(input_col = "word_list", output_col = "wo_stop_words")


### Creating a word2vec model
# To train the model we need a numeric representation of the sentence that can be passed to the 
# Logistic Regression classifier model. This as know as word embedding and the process we're
# using here is the built in Spark [Word2Vec](https://spark.rstudio.com/reference/ft_word2vec/)
# function. 

# _Note:_ If your model has already been saved, you can bypass this process by commenting out the following code:
# _Comment from here:_
# ============
w2v_model <- ft_word2vec(sc,
                        input_col = "wo_stop_words",
                        output_col = "result",
                        min_count = 5,
                        max_iter = 25,
                        vector_size = 400,
                        step_size = 0.0125
                       )

w2v_model_fitted <- ml_fit(w2v_model,sentence_values_tokenized)

ml_save(
  w2v_model_fitted,
  paste(Sys.getenv("STORAGE"),"/datalake/data/sentiment/w2v_model_fitted",sep=""),
  overwrite = TRUE
)
# =============
# _to here.
# 
# _And uncomment the lines below from:_
# ==============
# w2v_model_fitted <- ml_load(
#   sc, 
#   paste(Sys.getenv("STORAGE"),"/datalake/data/sentiment/w2v_model_fitted",sep="")
# )
# ==============
# _to here._


# `word2vec` is a transformer and will create a new column with a numeric representation of each 
# sentence. The data set is split into a test and training set for later validation.

w2v_transformed <- ml_transform(w2v_model_fitted, sentence_values_tokenized)

w2v_transformed_split <- w2v_transformed %>% sdf_random_split(training=0.7, test = 0.3)

### Creating a Logistic Regression model
# The next step is to train a logistic regression model using the `sent_score` binary label calculated earlier
# and the word2vec numeric representation calcuated in the previous step.
# 
# _Note:_ If your model has already been saved, you can bypass this process by commenting out the following code:
# _Comment from here:_
# ============
lr_model <- w2v_transformed_split$training %>% select(result,sent_score) %>% 
  ml_logistic_regression(
    sent_score ~ result,
    max_iter=500, 
    elastic_net_param=0.0,
    reg_param = 0.01
  )

ml_save(
   lr_model,
   paste(Sys.getenv("STORAGE"),"/datalake/data/sentiment/lr_model",sep=""),
   overwrite = TRUE
)
# =============
# _to here.
# 
# _And load the model by uncommenting the lines below from here:_
# ==============
# lr_model <- ml_load(
#   sc, 
#   paste(Sys.getenv("STORAGE"),"/datalake/data/sentiment/lr_model",sep="")
# )
# ==============
# _to here._

### Showing the Model Performance
# The model performance can be shown using the `ml_binary_classification_evaluator`
# function from sparklyr. 

pred_lr_training <- ml_predict(lr_model, w2v_transformed_split$training)

pred_lr_test<- ml_predict(lr_model, w2v_transformed_split$test)

ml_binary_classification_evaluator(pred_lr_training,label_col = "sent_score",
                        prediction_col = "prediction", metric_name = "areaUnderROC")

ml_binary_classification_evaluator(pred_lr_test,label_col = "sent_score",
                        prediction_col = "prediction", metric_name = "areaUnderROC")

# 89% seems resonable.