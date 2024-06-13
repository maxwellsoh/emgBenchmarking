echo "Retrieving UCI..."
wget -c -np https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip
unzip emg+data+for+gestures.zip 
rm emg+data+for+gestures.zip 
mv EMG_data_for_gestures-master/ uciEMG

# fix dataset by appending 0 to last time step for user 34 session 1
echo " 0" >> uciEMG/34/1_raw_data_10-51_07.04.16.txt

echo "Done retrieving UCI."
