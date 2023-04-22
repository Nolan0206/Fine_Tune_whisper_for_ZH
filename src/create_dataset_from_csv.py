# build a dataset from CSVs

from datasets import load_dataset, DatasetDict, Dataset, Audio
import sys


import pandas as pd

from tqdm import tqdm

from src.utils import data_split


# all paths without / at the end

# Update the path to read audio
def update_path(path):
    path_retain = path.split('/')[-3:]
    path_retain_str = '/'.join(path_retain)
    new_path = f"/source/DataRepository/{path_retain_str}"
    return new_path


def judge(list_):
    try:
        if not list_:
            raise IndexError("The list passed in is empty")
    except IndexError as error:
        print("引发异常：", repr(error))
        return False
    else:
        return True


def create_dataset(csv_list, ratio_train):
    if not judge(csv_list):
        sys.exit()
    audio_list, sentence_list = [], []
    for file in tqdm(csv_list, position=0, desc="file", leave=False, colour='green'):
        csv_ds = pd.read_csv(file, sep="\t")
        for index in range(csv_ds.shape[0]):
            audio_list.append(update_path(csv_ds.audio[index]))
            sentence_list.append(csv_ds.sentence[index])
    audio_train, audio_val = data_split(audio_list, ratio=float(ratio_train), shuffle=True)
    sentence_train, sentence_val = data_split(sentence_list, ratio=float(ratio_train), shuffle=True)
    train_dataset = Dataset.from_dict({"audio": audio_train, "sentence": sentence_train}).cast_column("audio", Audio(
        sampling_rate=16000))
    val_dataset = Dataset.from_dict({"audio": audio_val, "sentence": sentence_val}).cast_column("audio", Audio(
        sampling_rate=16000))

    dataset = DatasetDict()
    dataset['train'] = train_dataset
    dataset['val'] = val_dataset
    return dataset
    # return train_dataset, val_dataset



