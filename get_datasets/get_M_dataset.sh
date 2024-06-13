echo "Retrieving M_dataset..."
wget -c -np https://github.com/rajkundu/myoband/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
mv myoband-master/EvaluationDataset . # move out the EvaluationDataset 
rm -r myoband-master # delete everything else
mv EvaluationDataset M_dataset # rename to M_dataset
echo "Done retrieving M_dataset."
