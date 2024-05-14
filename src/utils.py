import numpy as np
import datetime
import json



def summarize_vocabulary(vocabulary: dict):
    frequencies = [item for item in vocabulary.items()]
    frequencies.sort(key=lambda item: item[1])
    frequencies = np.array(frequencies)
    return {
        "minimum": np.min(frequencies[:, 1].astype(int)),
        "maximum": np.max(frequencies[:, 1].astype(int)),
        "mean": np.mean(frequencies[:, 1].astype(int)),
        "median": np.median(frequencies[:, 1].astype(int)),
        "total": np.sum(frequencies[:, 1].astype(int)),
        "length": len(frequencies),
        "top_10": frequencies[-10:]
    }


def get_timestamp(date=None):
    if date is None: date = datetime.datetime.now()
    return f"{date:%Y%m%d%H%M}"

def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)
