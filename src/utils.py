import os
import sys
import numpy as np
import pandas as pd
import dill

from src.logger import logging
from src.exception import CustomException


def save_object(filename, obj):
    """Save object to file."""
    try:
        dir_path = os.path.dirname(filename)
        os.makedirs(dir_path, exist_ok=True)
        with open(filename, "wb") as file:
            dill.dump(obj, file)
        logging.info("Saved object to file")
    except Exception as e:
        raise CustomException(e, sys)
