mkdir -p NinaproDB2
echo "Retrieving NinaproDB2..."
for i in {1..40}
do 
    wget -c -np https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s${i}.zip -O ./NinaproDB2/s${i}.zip
    unzip ./NinaproDB2/s${i}.zip -d ./NinaproDB2
    rm ./NinaproDB2/s${i}.zip
done
echo "Done retrieving NinaproDB2."