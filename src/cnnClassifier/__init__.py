import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),#writes logs to the file
        logging.StreamHandler(sys.stdout)   #prints logs to the console
    ]
)

logger = logging.getLogger("cnnClassifierLogger")  #Creates a logger object named "cnnClassifierLogger" for consistent logging.