""" Finetuning example.
"""
from __future__ import print_function
import sys
import tempfile
import uuid
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import json

from deepmoji.model_def import deepmoji_transfer
from deepmoji.finetuning import (
    load_benchmark,
    finetune)
from deepmoji.class_avg_finetuning import class_avg_finetune
from keras.models import load_model

from deepmoji.finetuning import freeze_layers, chain_thaw, tune_trainable
from keras.optimizers import Adam


def finetune_special(model, texts, labels, nb_classes, batch_size, method,
             metric='acc', epoch_size=5000, nb_epochs=1000,
             error_checking=True, verbose=1):

    WEIGHTS_DIR = tempfile.mkdtemp()
    (X_train, y_train) = (texts[0], labels[0])
    (X_val, y_val) = (texts[1], labels[1])
    (X_test, y_test) = (texts[2], labels[2])

    checkpoint_path = '{}/deepmoji-checkpoint-{}.hdf5' \
                      .format(WEIGHTS_DIR, str(uuid.uuid4()))


    lr = 0.0001
    loss = 'binary_crossentropy' if nb_classes <= 2 \
        else 'categorical_crossentropy'


    result = chain_thaw(model, nb_classes=nb_classes,
                        train=(X_train, y_train),
                        val=(X_val, y_val),
                        test=(X_test, y_test),
                        batch_size=batch_size, loss=loss,
                        epoch_size=epoch_size,
                        nb_epochs=nb_epochs,
                        checkpoint_weight_path=checkpoint_path,
                        evaluate=metric, verbose=verbose)

    return model, result


ROOT_PATH = dirname(dirname(abspath(__file__)))
VOCAB_PATH = '{}/model/vocabulary.json'.format(ROOT_PATH)
PRETRAINED_PATH = '{}/model/deepmoji_weights.hdf5'.format(ROOT_PATH)
RESULTS_DIR = 'results'

training_parameters = [
    ('SS-Twitter', '../data/SS-Twitter/raw.pickle', 2, False),
    ('SS-Youtube', '../data/SS-Youtube/raw.pickle', 2, False),
]

# method = ['new', 'last', 'full', 'chain-thaw']
method = ['chain-thaw']

VERBOSE = 1
nb_tokens = 50000
nb_epochs = 1000
epoch_size = 1000

with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)


extend_with = 10000

# load the prained dataset
weight_path = PRETRAINED_PATH
nb_model_classes = 2

# Load first dataset
data = load_benchmark(training_parameters[0][1], vocab, extend_with=extend_with)

model = deepmoji_transfer(nb_model_classes, data['maxlen'], weight_path, extend_embedding=data['added'])

for FINETUNE_METHOD in method:
    for training_parameter in training_parameters:
        try:
            assert len(vocab) == nb_tokens

            dataset = training_parameter[0]
            dataset_path = training_parameter[1]
            nb_classes = training_parameter[2]
            use_f1 = training_parameter[3]

            data = load_benchmark(dataset_path, vocab, extend_with=extend_with)
            (X_train, y_train) = (data['texts'][0], data['labels'][0])
            (X_val, y_val) = (data['texts'][1], data['labels'][1])
            (X_test, y_test) = (data['texts'][2], data['labels'][2])


            # fintune the model
            # then generate result files
            print('Training: {}'.format(dataset_path))
            model.summary()
            model, result = finetune_special(model, data['texts'], data['labels'], nb_classes, data['batch_size'],
                                     FINETUNE_METHOD, metric='acc', verbose=VERBOSE)
            print('Test accuracy (dset = {}): {}'.format(dataset, result))


        except Exception, e:
            print('error in training ')
            print(e.message)


# save model
model.save('model_training_all_together.h5')


