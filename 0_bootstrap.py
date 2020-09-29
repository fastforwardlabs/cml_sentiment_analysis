## Part 0: Bootstrap File
# You need to at the start of the project. It will install the requirements, creates the 
# `STORAGE` environment variable and copy the data from 
# the `data` directory to into /datalake/data/sentiment of the `STORAGE`
# location.

# The STORAGE environment variable is the Cloud Storage location used by the DataLake 
# to store hive data. On AWS it will s3a://[something], on Azure it will be 
# abfs://[something] and on CDSW cluster, it will be hdfs://[something]

# Install the requirements for Python
!pip3 install --progress-bar off tensorflow==2.2.0
!pip3 install git+https://github.com/fastforwardlabs/cmlbootstrap#egg=cmlbootstrap

# Install the requirements for R. 
!R -e 'print("Test")'
!Rscript "R Code/0_install.R"

# Create the directories and upload data

from cmlbootstrap import CMLBootstrap
from IPython.display import Javascript, HTML
import os
import time
import json
import requests
import xml.etree.ElementTree as ET

# Set the setup variables needed by CMLBootstrap
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

# Set the STORAGE environment variable
try : 
  storage=os.environ["STORAGE"]
except:
  if os.path.exists("/etc/hadoop/conf/hive-site.xml"):
    tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
    root = tree.getroot()
    for prop in root.findall('property'):
      if prop.find('name').text == "hive.metastore.warehouse.dir":
        storage = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  else:
    storage = "/user/" + os.getenv("HADOOP_USER_NAME")
  storage_environment_params = {"STORAGE":storage}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = storage

# Upload the data to the cloud storage
!hdfs dfs -mkdir -p $STORAGE/datalake
!hdfs dfs -mkdir -p $STORAGE/datalake/data
!hdfs dfs -mkdir -p $STORAGE/datalake/data/sentiment
!hdfs dfs -copyFromLocal /home/cdsw/data/* $STORAGE/datalake/data/sentiment/
!hdfs dfs -copyFromLocal /home/cdsw/models.tgz $STORAGE/datalake/data/sentiment/

# Unpack nad move the models into the right directory
!mkdir /home/cdsw/models
!mkdir /home/cdsw/temp_data
!cp /home/cdsw/models.tgz /home/cdsw/models/
!cd /home/cdsw/models && tar xjvf models.tgz
!rm /home/cdsw/models/models.tgz
