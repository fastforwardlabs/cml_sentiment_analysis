if [ -z "$PYTHONMODEL" ]
then
    echo "R Model"
    Rscript "/home/cdsw/R Code/0_install.R"
else
    echo "Installing Python Requirements"
    pip3 install --progress-bar off tensorflow==2.2.0
    hdfs dfs -copyToLocal $STORAGE/datalake/data/sentiment/models.tgz
    mkdir models
    cp models.tgz models/
    cd models && tar xjvf models.tgz
    cd ../
fi
