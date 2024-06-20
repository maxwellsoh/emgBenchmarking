echo "Retrieving Hyser..."
for i in $(seq 1 9);
do
    echo "running"
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/pr_dataset/subject0${i}_session1/
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/pr_dataset/subject0${i}_session2/
done
for i in $(seq 10 20);
do
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/pr_dataset/subject${i}_session1/
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/pr_dataset/subject${i}_session2/
done

# rename folder 
mv physionet.org/files/hd-semg/1.0.0/pr_dataset/ . 
rm -r physionet.org/
mv pr_dataset/ hyser
echo "Done retrieving Hyser."
