import configparser
import numpy as np
import torch

from pipelineManager import PipeLineManager
import random
from utils import build_result_filename, open_result_file, close_result_file

np.random.seed(12)
random.seed(12)

torch.manual_seed(12)
torch.cuda.manual_seed_all(12)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('configuration.conf')
    configuration = config['SETTINGS']
    dsConf = config['DATASET']
    result_file_name = build_result_filename(configuration)
    open_result_file(result_file_name)

    pManager = PipeLineManager(configuration, dsConf)
    pManager.runPipeline()
    close_result_file()
