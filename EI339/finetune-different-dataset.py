""" Finetuning example.
"""
from __future__ import print_function
import sys
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import json

from deepmoji.model_def import deepmoji_transfer
from deepmoji.finetuning import (
    load_benchmark,
    finetune)
from deepmoji.class_avg_finetuning import class_avg_finetune
from keras.models import load_model


ROOT_PATH = dirname(dirname(abspath(__file__)))
VOCAB_PATH = '{}/model/vocabulary.json'.format(ROOT_PATH)
PRETRAINED_PATH = '{}/model/deepmoji_weights.hdf5'.format(ROOT_PATH)
RESULTS_DIR = 'results'

training_parameters = [
    ('SE0714', '../data/SE0714/raw.pickle', 3, True),
    ('Olympic', '../data/Olympic/raw.pickle', 4, True),
    ('PsychExp', '../data/PsychExp/raw.pickle', 7, True),
    ('SS-Twitter', '../data/SS-Twitter/raw.pickle', 2, False),
    ('SS-Youtube', '../data/SS-Youtube/raw.pickle', 2, False),
    ('SCv2-GEN', '../data/SCv2-GEN/raw.pickle', 2, False),

]

method = ['new', 'last', 'full', 'chain-thaw']

VERBOSE = 1
nb_tokens = 50000
nb_epochs = 1000
epoch_size = 1000

with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

for FINETUNE_METHOD in method:
    if FINETUNE_METHOD != 'chain-thaw':
        continue
    for i in range(1):
        for training_parameter in training_parameters:
            try:
                # debugging
                assert len(vocab) == nb_tokens

                dataset = training_parameter[0]
                dataset_path = training_parameter[1]
                nb_classes = training_parameter[2]
                use_f1 = training_parameter[3]

                if FINETUNE_METHOD == 'last':
                    extend_with = 0
                elif FINETUNE_METHOD in ['new', 'full', 'chain-thaw']:
                    extend_with = 10000
                else:
                    raise ValueError('Finetuning method not recognised!')

                # Load dataset and split it into (train, validate, test)
                data = load_benchmark(dataset_path, vocab, extend_with=extend_with)
                (X_train, y_train) = (data['texts'][0], data['labels'][0])
                (X_val, y_val) = (data['texts'][1], data['labels'][1])
                (X_test, y_test) = (data['texts'][2], data['labels'][2])

                # load the prained dataset
                weight_path = PRETRAINED_PATH if FINETUNE_METHOD != 'new' else None
                nb_model_classes = 2 if use_f1 else nb_classes
                model = deepmoji_transfer(nb_model_classes, data['maxlen'], weight_path, extend_embedding=data['added'])
                model.summary()

                # fintune the model and run the benchmark
                # then generate result files
                print('Training: {}'.format(dataset_path))
                if use_f1:
                    model, result = class_avg_finetune(model, data['texts'], data['labels'], nb_classes,
                                                       data['batch_size'], FINETUNE_METHOD, verbose=VERBOSE)
                    print('Overall F1 score (dset = {}): {}'.format(dataset, result))
                    with open('{}/{}_{}_{}_f1_results.txt'.
                                      format(RESULTS_DIR, dataset, FINETUNE_METHOD, i),
                              "w") as f:
                        f.write("F1: {}\n".format(result))

                else:
                    model, result = finetune(model, data['texts'], data['labels'], nb_classes, data['batch_size'],
                                             FINETUNE_METHOD, metric='acc', verbose=VERBOSE)
                    print('Test accuracy (dset = {}): {}'.format(dataset, result))
                    with open('{}/{}_{}_{}_acc_results.txt'.
                                      format(RESULTS_DIR, dataset, FINETUNE_METHOD, i),
                              "w") as f:
                        f.write("Acc: {}\n".format(result))

                # save model
                model.save('{}_{}_model.h5'.format(dataset, FINETUNE_METHOD))

            except Exception, e:
                print('error in training, dataset: {}, method: {}, iteration: {}'.format(dataset, FINETUNE_METHOD, i))
                print(e.message)




