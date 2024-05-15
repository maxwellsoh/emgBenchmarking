
mkdir -p NinaproDB5
echo "Retrieving NinaproDB5..."


for i in $(seq 1 10);
do 
    wget -c -np https://ninapro.hevs.ch/files/DB5_Preproc/s${i}.zip -O ./NinaproDB5/s${i}.zip
    unzip ./NinaproDB5/s${i}.zip -d ./NinaproDB5
    rm ./NinaproDB5/s${i}.zip
done
echo "Done retrieving NinaproDB5."