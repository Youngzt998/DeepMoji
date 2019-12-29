import csv
import os


header = [ 'dataset', 'new', 'last', 'full', 'chain-thaw' ]
rows = []
scores = {

}

resList = os.listdir('results')


for resFile in resList:
    if not resFile.endswith('results.txt'):
        continue
    dataset, method, _, scoreType, _ = resFile.split("_")
    print scoreType
    if scoreType != 'acc':
        continue
    scores[dataset] = {}


for resFile in resList:
    if not resFile.endswith('.txt'):
        continue
    dataset, method, _, scoreType, _ = resFile.split("_")

    if scoreType != 'acc':
        continue

    with open('./results/{}'.format(resFile)) as f:
        _, score = (f.read()).split(' ')
        score, _ = score.split('\n')
    scores[dataset][method] = score

print scores


def getScore(dataset, method):
    if not scores.has_key(dataset):
        return ''

    if not scores[dataset].has_key(method):
        return ''

    return scores[dataset][method]


for dataset in scores:
    rows.append([dataset, getScore(dataset, 'new'), getScore(dataset, 'last'), getScore(dataset, 'full'), getScore(dataset, 'chain-thaw')])

print rows


with open('./results/result_acc.csv', 'w') as f:
    ff = csv.writer(f)
    ff.writerow(header)
    ff.writerows(rows)
