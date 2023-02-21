from torchaudio.datasets import SPEECHCOMMANDS
from preprocess import PreProcessor
import os
import matplotlib.pyplot as plt
import numpy as np


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, data_dir, subset: str = None):
        super().__init__(data_dir, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
    
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def main():
    train_set = SubsetSC("../data/", "training")
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[38567]
    print(label)

    preprocessor = PreProcessor(dataset_type="torch_dataset")
    output_data = preprocessor.create_spectrogram(waveform)

    plt.plot(output_data)

    return output_data


if __name__ == "__main__":
    main()