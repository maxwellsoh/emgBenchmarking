echo "Retrieving UCI..."
wget -c -np https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip
unzip emg+data+for+gestures.zip 
rm emg+data+for+gestures.zip 
mv EMG_data_for_gestures-master/ uciEMG
echo "Done retrieving UCI."
