library(sparklyr)
library(dplyr)
library(jsonlite)

storage <- Sys.getenv("STORAGE")

config <- spark_config()
config$spark.hadoop.yarn.resourcemanager.principal <- Sys.getenv("HADOOP_USER_NAME")
config$spark.yarn.access.hadoopFileSystems <- storage
config$spark.driver.memory <- "4g"
sc <- spark_connect(master = "local", config=config)

w2v_model_fitted <- ml_load(
  sc, 
  paste(Sys.getenv("STORAGE"),"/datalake/data/sentiment/w2v_model_fitted",sep="")
)

lr_model <- ml_load(
  sc, 
  paste(Sys.getenv("STORAGE"),"/datalake/data/sentiment/lr_model",sep="")
)


#args <- fromJSON('{"sentence":"Im no dunce. I was born an oaf and Ill die an oaf"}')

predict_sentiment <- function(args) {
  test_text_df <- as.data.frame(args$sentence)
  colnames(test_text_df) <- "spoken_words"

  sdf_copy_to(sc, test_text_df, name="test_text", overwrite = TRUE)
  test_text <- tbl(sc, "test_text")
  
  test_text <- test_text %>%
    mutate(spoken_words = regexp_replace(spoken_words, "[_\"\'():;,.!?\\-]", " ")) %>%
    ft_tokenizer(input_col="spoken_words",output_col= "word_list") %>%
    ft_stop_words_remover(input_col = "word_list", output_col = "wo_stop_words")
  
  test_text <- ml_transform(w2v_model_fitted,test_text)

  result <- ml_transform(lr_model,test_text)
  result <- as.data.frame(result)
  
  if(result$prediction==0) {
    sentiment <- 'Negative'
    conf <- round(result$probability[[1]][1] * 100,3)
  }
  else{
    sentiment = 'Positive'  
    conf = round(result$probability[[1]][2] * 100,3)
  }
  return(list("sentiment" = sentiment, "confidence" = conf))
}
