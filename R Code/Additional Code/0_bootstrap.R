# This file does the same thing as the 0_bootstrap.py file in the home director of this project. This is here in R so that you
# use parts of this code in other CDSW/CML projects if needed. 

install.packages("dplyr")
install.packages("tibble")
install.packages("sparklyr")
install.packages("ggthemes")

library(xml2)
library(httr)

HOST <- paste(strsplit(Sys.getenv("CDSW_API_URL"),"/")[[1]][1],"//",Sys.getenv("CDSW_DOMAIN"),sep="")
USERNAME <- strsplit(Sys.getenv("CDSW_PROJECT_URL"),"/")[[1]][7]
API_KEY = Sys.getenv("CDSW_API_KEY") 
PROJECT_NAME = Sys.getenv("CDSW_PROJECT")  

if (Sys.getenv("STORAGE") == "") {
  hive_config <- read_xml('/etc/hadoop/conf/hive-site.xml')
  hive_props <- xml_find_all(hive_config,"property")
  warehouse_prop <- hive_props[xml_text(xml_find_all(hive_props, "name")) == "hive.metastore.warehouse.external.dir"]
  warehouse_dir <- xml_text(xml_find_first(warehouse_prop, "value"))
  storage <- paste(strsplit(as.character(warehouse_dir),"/")[[1]][1],"//",strsplit(as.character(warehouse_dir),"/")[[1]][3],sep="")

  create_env_url <- paste(HOST,"api/v1/projects",USERNAME,PROJECT_NAME,"environment",sep="/")

  r <- PUT(
    create_env_url, 
    body=list(STORAGE=storage),
    encode="json",
    authenticate(API_KEY,"")
  )
  r

} else {
  storage <- Sys.getenv("STORAGE")
}

system(paste("hdfs dfs -mkdir ",storage,"/datalake",sep=""))
system(paste("hdfs dfs -mkdir ",storage,"/datalake/data",sep=""))
system(paste("hdfs dfs -mkdir ",storage,"/datalake/data/sentiment",sep=""))
system(paste("hdfs dfs -copyFromLocal simpsons_dataset.csv ",storage,"/datalake/data/sentiment/simpsons_dataset.csv",sep=""))
system(paste("hdfs dfs -copyFromLocal AFINN-en-165.txt ",storage,"/datalake/data/sentiment/AFINN-en-165.txt",sep=""))
