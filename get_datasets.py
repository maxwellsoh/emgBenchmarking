# Inspiration: https://stackoverflow.com/questions/62662480/using-python-requests-to-download-multiple-zip-files-from-links

from bs4 import BeautifulSoup 
import zipfile
import requests
import re
import os
import argparse
import process_NinaproDB5


def get_soup(url):
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    zip_links = soup.findAll("a", attrs={'href': re.compile(".zip")})
    return zip_links

def download(url, folder_name):
    zip_links = get_soup(url)

    # create new folder
    parent_dir = os.getcwd() 
    new_dir = os.path.join(parent_dir, folder_name)
    os.mkdir(new_dir)

    for link in zip_links:
        file_link = link.get('href')
        
        # create new directory adress
        final_path = os.path.join(new_dir, link.text)

        # TODO: error catch if folder already exists

        # write the zip folder
        with open(final_path, 'wb') as file:
            response = requests.get(url + file_link)
            file.write(response.content) 

        # unzip the folder 
        with zipfile.ZipFile(final_path, 'r') as zip_ref:
            zip_ref.extractall(new_dir)

        # TODO: delete the old zip folder

        print("Downloaded and extracted:", file_link)


def get_DB5():
    print("Retrieving NinaproDB5...")
    url = "https://ninapro.hevs.ch/files/DB5_Preproc/"
    folder_name = "NinaproDB5"
    download(url, folder_name)
    print("Downloaded NinaproDB5")
    print("Processing NinaproDB5...")
    process_NinaproDB5.get()
    print("Done processing Ninapro DB5")

def get_DB2():
    print("Retrieving NinaproDB2...")
    url = "https://ninapro.hevs.ch/files/DB2_Preproc/"
    folder_name = "NinaproDB2"
    download(url, folder_name)
    print("Done retrieving NinaproDB2")


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--NinaproDB5', help="Creates NinaproDB5 folder", action='store_true')
    parser.add_argument('--NinaproDB2', help="Creates NinaproDB2 folder", required=False)    

    args = parser.parse_args()

    if args.NinaproDB5:
        get_DB5()

    if args.NinaproDB2:
        get_DB2()


main()

    


