echo "Retrieving MCS_EMG..."
wget -np https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ckwc76xr2z-2.zip
unzip ckwc76xr2z-2.zip 
rm ckwc76xr2z-2.zip
mv ckwc76xr2z-2 MCS_EMG
echo "Done retrieving MCS_EMG."