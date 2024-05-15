for i in {1..9}
do
    echo "running"
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/pr_dataset/subject0${i}_session1/
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/pr_dataset/subject0${i}_session2/
done
for i in {10..20}
do
    echo "2nd condition"
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/pr_dataset/subject${i}_session1/
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/pr_dataset/subject${i}_session2/
done