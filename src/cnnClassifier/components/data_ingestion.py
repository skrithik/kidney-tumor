import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import (DataIngestionConfig)



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL#Reads dataset URL and target file path from config
            zip_download_dir = self.config.local_data_file#zip_download_dir holds the full path where the downloaded dataset zip file will be saved
            os.makedirs("artifacts/data_ingestion", exist_ok=True)#Ensures folder artifacts/data_ingestion/ exists.
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")#Logs that download is starting.

            file_id = dataset_url.split("/")[-2]#Extracts the file ID from the Google Drive URL.
            #example
            #dataset_url.split("/") splits the string by / into a list:
            #['https:', '', 'drive.google.com', 'file', 'd', '1dkE79W8XToiv0IesetUyDNmTUQLehSla', 'view?usp=sharing']
            #[-2] picks the second-to-last element, which is:
            #'1dkE79W8XToiv0IesetUyDNmTUQLehSla'-This is the Google Drive file ID
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)#gdown.download() to download the zip file.

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir#Reads target unzip directory from config.
        os.makedirs(unzip_path, exist_ok=True)#Ensures the unzip directory exists.
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:#Opens the downloaded zip file.
            zip_ref.extractall(unzip_path)#Extracts all contents into the target directory.