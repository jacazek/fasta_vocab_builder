import os
import shutil
from multiprocessing import Process, Queue
from fasta_utils.vocab import Vocab
from fasta_utils import FastaFileReader
import utils
from fasta_utils.tokenizers import KmerTokenizer
import tqdm
import mlflow
import argparse
from dataclasses import dataclass
from typing import Optional
import numpy as np
from collections import Counter

script_directory = os.path.dirname(os.path.realpath(__file__))


def get_absolute_path(path):
    return os.path.abspath(os.path.join(script_directory, path))


@dataclass
class BuildArgs:
    kmer_size: int
    stride: int
    fasta_directory: int
    number_of_workers: Optional[int] = None


class TokenStore():
    def __init__(self):
        # self.tokens = Counter
        self.tokens = {}

    def add(self, token):
        # self.tokens.update(token)
        # self.tokens.setdefault(token, 0)
        # self.tokens[token] += 1
        try:
            self.tokens[token] = self.tokens[token] + 1
        except KeyError:
            self.tokens[token] = 1
        # if token in self.tokens:
        #     self.tokens[token] += 1
        # else:
        #     self.tokens[token] = 1

    def merge(self, source_token_store):
        for token, value in source_token_store.tokens.items():
            if token in self.tokens:
                self.tokens[token] = self.tokens[token] + value
            else:
                self.tokens[token] = value


def producer(job_list, t, tokenizer, queue):
    for index, job in enumerate(job_list):
        file_name, fasta_file = job
        name = f"{file_name} ({index+1} of {len(job_list)})"
        t.set_description(name)
        with FastaFileReader(fasta_file) as fasta_file_reader:
            for header, sequence in fasta_file_reader.read_all():
                token_store = TokenStore()
                t.set_description(f"{name}:{header}")
                for token, reversed_token in tokenizer.tokenize(sequence):
                    token_store.add(token)
                    token_store.add(reversed_token)
                    t.update()
                queue.put(token_store)


def consumer(queue, index):
    token_store = TokenStore()
    t = tqdm.tqdm(desc="vocab_builder", position=index)
    while True:
        tokens_from_producer = queue.get(block=True)
        if tokens_from_producer is None:
            break
        token_store.merge(tokens_from_producer)
        t.update(sum(tokens_from_producer.tokens.values()))

    queue.put(token_store)
    queue.put(None)
    t.close()


def main(build_args: BuildArgs):
    fasta_file_directory = get_absolute_path(build_args.fasta_directory)
    fasta_file_extension = ".fa.gz"

    fasta_files = [(file, f"{fasta_file_directory}/{file}") for file in os.listdir(fasta_file_directory) if
                   file.endswith(fasta_file_extension)]
    if (len(fasta_files)) <= 0:
        raise Exception("No fasta files located")

    # Start the experiment now that we know we have files
    mlflow.set_tracking_uri(uri="http://localhost:8080")
    experiment = mlflow.get_experiment_by_name("Fasta Vocabulary")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        token_queue = Queue()
        tokenizer = KmerTokenizer(build_args.kmer_size, build_args.stride, include_compliement=True)
        producer_progress_bars = []
        producer_processes = []


        # Switch to using the process pool executor
        number_of_workers = len(fasta_files)

        if build_args.number_of_workers is not None:
            number_of_workers = min(len(fasta_files), build_args.number_of_workers)

        # capture parameters for the run
        vocabulary_parameters = {
            "kmer_size": build_args.kmer_size,
            "stride": build_args.stride,
            "number_files_to_process": len(fasta_files),
            "tokenizer": tokenizer.get_id_string(),
            "fasta_directory": build_args.fasta_directory,
            "number_of_workers": number_of_workers
        }
        mlflow.log_params(vocabulary_parameters)

        # divide the files by worker and schedule the jobs
        jobs = np.array_split(fasta_files, number_of_workers)
        for index, job_list in enumerate(jobs):
            # name, fasta_file = fasta_files[0]
            t = tqdm.tqdm(position=index)
            producer_progress_bars.append(t)
            producer_processes.append(Process(target=producer, args=(job_list, t, tokenizer, token_queue)))

        consumer_process = Process(target=consumer, args=(token_queue, number_of_workers))

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

        # gather metrics and metadata
        vocabulary_metadata = {
                                  "id": vocabulary_id,
                                  "included_sequences": [os.path.basename(file) for file, path in fasta_files]
                              } | vocabulary_parameters | summary

        metrics = summary.copy()
        del metrics["top_10"]

        # log the metrics
        mlflow.log_metrics(metrics)

        # Create artifacts from the run
        os.mkdir(get_absolute_path(f"../artifacts/{vocabulary_id}"))
        metadata_file = get_absolute_path(f"../artifacts/{vocabulary_id}/{vocabulary_metadata_file}")
        frequency_file = get_absolute_path(f"../artifacts/{vocabulary_id}/{vocabulary_frequency_file}")
        pickle_file = get_absolute_path(f"../artifacts/{vocabulary_id}/{vocabulary_file}")
        utils.save_json(vocabulary_metadata, metadata_file)
        utils.save_json(frequencies, frequency_file)
        Vocab.save(vocabulary, pickle_file)

        # capture the artifacts
        mlflow.log_artifacts(get_absolute_path(f"../artifacts/{vocabulary_id}/"))
        shutil.rmtree(get_absolute_path(f"../artifacts/{vocabulary_id}/"))


if __name__ == "__main__":
    default_data_directory = os.path.abspath(os.path.join(script_directory, "../data"))
    print(script_directory)
    parser = argparse.ArgumentParser(description="Train kmer vocabulary.")
    parser.add_argument("--kmer_size", type=int, default=7, help="The size of kmers for the vocabulary.")
    parser.add_argument("--stride", type=int, default=3, help="The stride between kmers.")
    parser.add_argument("--fasta_directory", type=str, default=default_data_directory,
                        help="The directory containing compressed fast files and fasta index files.")
    parser.add_argument("--number_of_workers", type=int, default=None,
                        help="The number of worker processes to build the vocabulary (1 process gets a single file).")

    args = parser.parse_args()
    build_args = BuildArgs(**vars(args))
    main(build_args)
