import numpy as np
import datetime
import json



def summarize_vocabulary(vocabulary: dict):
    frequencies = [item for item in vocabulary.items()]
    frequencies.sort(key=lambda item: item[1])
    frequencies = np.array(frequencies)
    return ({
        "maximum_kmer_frequencey": int(np.max(frequencies[:, 1].astype(int))),
        "mean_kmer_frequency": int(np.mean(frequencies[:, 1].astype(int))),
        "median_kmer_frequency": int(np.median(frequencies[:, 1].astype(int))),
        "total_kmers_processed": int(np.sum(frequencies[:, 1].astype(int))),
        "total_unique_kmers": len(frequencies),
        "top_10": frequencies[-10:].tolist()
    }, frequencies.tolist())


def get_timestamp(date=None):
    if date is None: date = datetime.datetime.now()
    return f"{date:%Y%m%d%H%M}"


def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)


translation_table = str.maketrans("ATCGN", "TAGCN")


def get_compliment(sequence):
    return sequence[::-1].translate(translation_table)

