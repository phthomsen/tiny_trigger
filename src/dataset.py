"""
This script takes advantage of the TensorFlow Lite Micro library to create the training data we need to train a model that runs on a
very constrained device. In this example it is an Arduino micro controller.

The original training script loads the entire dataset into memory. What we do here is transform the data, store it on disk and create 
a dataloader object to be more lazy and memory efficient.
"""

import tensorflow as tf
import sys
import os
try:
    sys.path.append("tensorflow/tensorflow/examples/speech_commands/")
    import input_data
    import models
except ModuleNotFoundError:
    logging.warning("tensorflow not found, need to clone it first")
    os.system("git clone https://github.com/tensorflow/tensorflow.git")
    import input_data
    import models
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger()


class FrankenSteinDataSet(Dataset):
    """
    Our dataset is created with tensorflow functionality, but will yield a pytorch Dataset to train a torch model.
    """
    def __init__(self, mode):
        PREPROCESS = 'micro'
        WINDOW_STRIDE = 20
        WANTED_WORDS = "yes,no"

        SAMPLE_RATE = 16000
        CLIP_DURATION_MS = 1000
        WINDOW_SIZE_MS = 30.0
        FEATURE_BIN_COUNT = 40
        BACKGROUND_FREQUENCY = 0.8
        BACKGROUND_VOLUME_RANGE = 0.1
        TIME_SHIFT_MS = 100

        DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        DATASET_DIR =  '../data/SpeechCommands/speech_commands_v0.02/'
        LOGS_DIR = 'logs/'

        VALIDATION_PERCENTAGE = 10
        TESTING_PERCENTAGE = 10

        # Calculate the percentage of 'silence' and 'unknown' training samples required
        # to ensure that we have equal number of samples for each label.
        number_of_labels = WANTED_WORDS.count(',') + 1
        number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
        equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))
        SILENT_PERCENTAGE = equal_percentage_of_training_samples
        UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples    
        
        logger.info("initializing the tensorflow Audio processor and process data.")
        model_settings = models.prepare_model_settings(
            len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
            SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
            WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
        
        audio_processor = input_data.AudioProcessor(
            DATA_URL, DATASET_DIR,
            SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
            WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
            TESTING_PERCENTAGE, model_settings, LOGS_DIR)
        
        with tf.compat.v1.Session() as sess:
            self._data, self._labels = audio_processor.get_data(
                how_many=-1,
                offset=0,
                model_settings=model_settings,
                background_frequency=BACKGROUND_FREQUENCY,
                background_volume_range=BACKGROUND_VOLUME_RANGE,
                time_shift=TIME_SHIFT_MS,
                mode=mode,
                sess=sess)

    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, idx):
        # this already gives us torch tensors
        return self._data[idx], self._labels[idx]


def main():
    dataset = FrankenSteinDataSet("training")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for data, label in loader:
        print(f"dtype of data is {type(data)}")
        print(f"shape of data is {data.shape}")
        print(f"data looks like this: {data}")

        break

    logger.info("DONE")

if __name__ == "__main__":
    main()