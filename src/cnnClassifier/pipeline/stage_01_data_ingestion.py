from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()#Reads config → Creates structured config object for data ingestion.
        data_ingestion_config = config.get_data_ingestion_config()#Initializes DataIngestion component with config.
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()#Downloads dataset zip.
        data_ingestion.extract_zip_file()#Extracts dataset zip.



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e