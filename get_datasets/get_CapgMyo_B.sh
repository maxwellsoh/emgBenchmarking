echo "Retrieving CapgMyo_B..."

# Should really only have to download dbb folder, gdown isn't working
gdown https://drive.google.com/drive/folders/1aEy2_5O4j7J7ls26EWdj95PGWBwr2hzK?usp=sharing -O ./CapgMyo_Main --folder
mv CapgMyo_Main/capg-dbb.zip . 
rm -r CapgMyo_Main 
unzip capg-dbb.zip
rm capg-dbb.zip 
mv capg-dbb/dbb . 
rm -r capg-dbb
ls

# I can manually sort by number, but there's no note in this data set about preprocessed vs not preprocessed 
destination_dir="./dbb"
for file in ./dbb/*; do 
    filename=$(basename "$file")
    aaa=$(echo "$filename" | cut -d'-' -f1 | cut -c1-3)
    mkdir -p "$destination_dir/dbb-preprocessed-$aaa"
    mv "$file" "$destination_dir/dbb-preprocessed-$aaa/$filename"

echo "Done retrieving CapgMyo_B."
done 



