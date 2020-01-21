

# sandbox for trying stuff out
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from scipy import sparse
import scipy

import math
from datetime import datetime

import json
import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score, f1_score

import numpy as np

with open('data/processed/papers.json') as f:
    papers = json.load(f)
    
with open('data/processed/people.json') as f:
    people = json.load(f)



# this doesn't help
def get_time(dt_string):
    x = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S.%f')
    hour = x.hour
    if hour < 4:
        return "wee_morning"
    elif hour < 8:
        return 'early_am'
    elif hour < 12:
        return 'am'
    elif hour < 16:
        return "early_pm"
    elif hour < 20:
        return 'pm'
    else:
        return 'night'

def get_order(idx, total):
    x = total / 3
    if idx < x:
        return "order_early"
    elif idx < (x * 2):
        return "order_mid"
    else:
        return "order_late"


labels = []
corpus = []
meta_covariates = []
for pid, paper_info in papers.items():
    winners = paper_info["winning_reviewers"]
    losers = paper_info["losing_reviewers"]
    if paper_info["review_consults"] is None:
        continue
    for i, consult in enumerate(paper_info["review_consults"]):
        text = consult["text"]
        author = consult["author"]
        author_info = people.get(author, None)
        if author_info is None:
            continue

        if author not in winners + losers:
            continue

        labels += [int(author in winners)]
        corpus += [text]

        time = get_time(consult['date'])
        gender = 'gender_%s' % people[consult['author']]['gender']

        try:
            university = '_'.join(people[consult['author']]['institution'].strip().split())
        except AttributeError:
            university = 'none'
        experience = math.log(len(people[consult['author']]['pids']))

        meta_covariates.append((time + ' ' + university + ' ' + gender + ' author_' + str(author), experience))



def run_regression(X, Y, feature_names):
    lr = LogisticRegression()

    cv = cross_validate(
        lr, X, Y, cv=10,
        scoring={
            'acc': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'auc': make_scorer(roc_auc_score),
        })

    fit = lr.fit(X, Y)
    features_values = sorted(
        zip(
            feature_names, 
            np.squeeze(fit.coef_)), 
        key=lambda x: x[1], reverse=True)
    print(features_values)

    print('F1: ', np.mean(cv['test_f1']))
    print('ACC: ', np.mean(cv['test_acc']))
    print('AUC: ', np.mean(cv['test_auc']))



#################################################################
# predict winner from email text
#
# vectorizer = CountVectorizer(max_features=1000, binary=True)
# vectorizer.fit(corpus)
# corpus = vectorizer.transform(corpus)
# run_regression(corpus, labels, vectorizer.get_feature_names())


#################################################################
# predict winner from institutino + experience + gender + time  + author id + order in chain
#
# bucket experience into quartiles
# X_meta, experience = zip(*meta_covariates)
# quantiles = np.percentile(experience, [25, 50, 75])
# experience = np.searchsorted(quantiles, experience)
# new_X_meta = []
# for a, b in zip(X_meta, experience):
#     new_X_meta.append(a + ' quartile' + str(b))

# vectorizer = CountVectorizer(max_features=500, binary=True)
# vectorizer.fit(new_X_meta)
# new_X_meta = vectorizer.transform(new_X_meta)
# experience = np.expand_dims(np.array(experience), axis=1)
# new_X_meta = np.concatenate((new_X_meta.toarray(), experience), axis=1)

# run_regression(new_X_meta, labels, vectorizer.get_feature_names() + ['experience'])



#################################################################
# predict winnner from both (text + metadata)
# 
vectorizer = CountVectorizer(max_features=1000, binary=True)
vectorizer.fit(corpus)
word_features = vectorizer.get_feature_names()
corpus = vectorizer.transform(corpus)

X_meta, experience = zip(*meta_covariates)
quantiles = np.percentile(experience, [25, 50, 75])
experience = np.searchsorted(quantiles, experience)
new_X_meta = []
for a, b in zip(X_meta, experience):
    new_X_meta.append(a + ' quartile' + str(b))

vectorizer = CountVectorizer(max_features=1000, binary=True)
vectorizer.fit(new_X_meta)
new_X_meta = vectorizer.transform(new_X_meta)
experience = np.expand_dims(np.array(experience), axis=1)
new_X_meta = np.concatenate((new_X_meta.toarray(), experience), axis=1)
meta_features = vectorizer.get_feature_names() + ['experience']

X_final = np.concatenate([corpus.toarray(), new_X_meta], axis=1)

run_regression(X_final, labels, word_features + meta_features)





