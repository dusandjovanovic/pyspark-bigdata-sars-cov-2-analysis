docker exec -it namenode hdfs dfs -test -e /data
if [ $? -eq 1 ]
then
  docker exec -it namenode hdfs dfs -mkdir /data
fi

docker exec -it namenode hdfs dfs -test -e /data/COVID.metadata.xlsx
if [ $? -eq 1 ]
then
  docker exec -it namenode hdfs dfs -copyFromLocal /data_part /data
fi