import os
import shutil
from multiprocessing import Process, Queue
from fasta_utils.vocab import Vocab
from fasta_utils import FastaFileReader
import utils
from fasta_utils.tokenizers import KmerTokenizer
import tqdm
import mlflow

mlflow.set_tracking_uri(uri="http://localhost:8080")

script_directory = os.path.dirname(os.path.realpath(__file__))


def get_absolute_path(path):
    return os.path.abspath(os.path.join(script_directory, path))


fasta_file_directory = get_absolute_path("../data")
fasta_file_extension = ".fa.gz"
kmer_min = 3
kmer_max = 7
stride = 3


class TokenStore():
    def __init__(self):
        self.tokens = {}

    def add(self, token):
        try:
            self.tokens[token] = self.tokens[token] + 1
        except KeyError:
            self.tokens[token] = 1

    def merge(self, source_token_store):
        for token, value in source_token_store.tokens.items():
            if token in self.tokens:
                self.tokens[token] = self.tokens[token] + value
            else:
                self.tokens[token] = value


fasta_files = [(file, f"{fasta_file_directory}/{file}") for file in os.listdir(fasta_file_directory) if
               file.endswith(fasta_file_extension)]
if (len(fasta_files)) <= 0:
    raise Exception("No fasta files located")


def producer(name, fasta_file, t, tokenizer, queue):
    with FastaFileReader(fasta_file) as fasta_file_reader:
        for header, sequence in fasta_file_reader.read_all():
            token_store = TokenStore()
            t.set_description(f"{name}:{header}")
            for token, reversed_token in tokenizer.tokenize(sequence):
                token_store.add(token)
                token_store.add(reversed_token)
                t.update()
            queue.put(token_store)


def consumer(queue):
    token_store = TokenStore()
    t = tqdm.tqdm(desc="vocab_builder", position=len(fasta_files))
    while True:
        tokens_from_producer = queue.get(block=True)
        if tokens_from_producer is None:
            break
        token_store.merge(tokens_from_producer)
        t.update(sum(tokens_from_producer.tokens.values()))

    queue.put(token_store)
    queue.put(None)
    t.close()

experiment = mlflow.get_experiment_by_name("Fasta Vocabulary")
with mlflow.start_run(experiment_id=experiment.experiment_id):
    token_queue = Queue()
    # tokenizer = VariableKmerTokenizer(kmer_min, kmer_max, stride)
    tokenizer = KmerTokenizer(kmer_max, stride, include_compliement=True)
    producer_progress_bars = []
    producer_processes = []

    # Switch to using the process pool executor
    for index, value in enumerate(fasta_files):
        name, fasta_file = value
        t = tqdm.tqdm(desc=f"{name}", position=index)
        producer_progress_bars.append(t)
        producer_processes.append(Process(target=producer, args=(name, fasta_file, t, tokenizer, token_queue)))

    consumer_process = Process(target=consumer, args=(token_queue,))

    for thread in producer_processes:
        thread.start()
    consumer_process.start()

    for thread in producer_processes:
        thread.join()

    for progress_bar in producer_progress_bars:
        progress_bar.close()

    token_queue.put(None)

    ### SKETCHY ####
    # could accidentally consume the None token we just placed on the queue
    # figure out a better way to handle this
    tokens = token_queue.get(block=True)
    consumer_process.join()

    summary, frequencies = utils.summarize_vocabulary(tokens.tokens)
    print(summary)

    special_tokens = ["pad", "unk"]

    vocabulary = Vocab(tokens.tokens, specials=special_tokens)
    vocabulary.set_default_index(vocabulary["unk"])

    vocabulary_id = f"{tokenizer.get_id_string()}-{utils.get_timestamp()}"
    vocabulary_file = f"{vocabulary_id}.pickle"
    vocabulary_metadata_file = f"{vocabulary_id}.json"
    vocabulary_frequency_file = f"{vocabulary_id}_frequencies.json"

    vocabulary_parameters = {
        "kmer_size": kmer_max,
        "stride": stride,
        "number_files_processed": len(fasta_files),
        "tokenizer": tokenizer.get_id_string()
    }

    vocabulary_metadata = {
                              "id": vocabulary_id,
                              "included_sequences": [os.path.basename(file) for file, path in fasta_files]
                          } | vocabulary_parameters | summary

    metrics = summary.copy()

    del metrics["top_10"]

    mlflow.log_params(vocabulary_parameters)
    mlflow.log_metrics(metrics)

    os.mkdir(get_absolute_path(f"../artifacts/{vocabulary_id}"))
    metadata_file = get_absolute_path(f"../artifacts/{vocabulary_id}/{vocabulary_metadata_file}")
    frequency_file = get_absolute_path(f"../artifacts/{vocabulary_id}/{vocabulary_frequency_file}")
    pickle_file = get_absolute_path(f"../artifacts/{vocabulary_id}/{vocabulary_file}")
    utils.save_json(vocabulary_metadata, metadata_file)
    utils.save_json(frequencies, frequency_file)
    Vocab.save(vocabulary, pickle_file)

    mlflow.log_artifacts(get_absolute_path(f"../artifacts/{vocabulary_id}/"))
    shutil.rmtree(get_absolute_path(f"../artifacts/{vocabulary_id}/"))



