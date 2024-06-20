echo "Retrieving MyoArmbandDataset..."
wget -c -np https://github.com/rajkundu/myoband/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
mv myoband-master/EvaluationDataset . # move out the EvaluationDataset 
rm -r myoband-master # delete everything else
mv EvaluationDataset myoarmbanddataset # rename to myoarmbanddataset
echo "Done retrieving MyoArmbandDataset."
