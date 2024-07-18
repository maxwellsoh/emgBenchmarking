mkdir -p NinaproDB3
echo "Retrieving NinaproDB3..."


for i in $(seq 1 11);
do 
    wget -c -np https://ninapro.hevs.ch/files/db3_Preproc/s${i}_0.zip -O ./NinaproDB3/s${i}.zip
    unzip ./NinaproDB3/s${i}.zip -d ./NinaproDB3
    rm ./NinaproDB3/s${i}.zip
done
echo "Done retrieving NinaproDB3."