from __future__ import print_function
import sys
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import json
from deepmoji.model_def import deepmoji_transfer
from deepmoji.finetuning import (
    load_benchmark,
    finetune)

ROOT_PATH = dirname(dirname(abspath(__file__)))
VOCAB_PATH = '{}/model/vocabulary.json'.format(ROOT_PATH)
PRETRAINED_PATH = None
DATASET_PATH = '../data/SS-Youtube/raw.pickle'
nb_classes = 2

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)
data = load_benchmark(DATASET_PATH, vocab)

model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH)
model.summary()
model, acc = finetune(model, data['texts'], data['labels'], nb_classes, data['batch_size'], method='new')

model.save('model-new-arch-test.h5')


print('Acc: {}'.format(acc))