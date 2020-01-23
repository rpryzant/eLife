# cleaning up 4 which worked

import pickle
from tqdm import tqdm
import json
import torch
import pandas as pd
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
import numpy as np
import time
import datetime
import random
from collections import defaultdict
import argparse
import os
import scipy
import sklearn
import math

CUDA = (torch.cuda.device_count() > 0)


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--papers",
    default="data/processed/papers.json",
    type=str,
    help="path to papers json"
)
parser.add_argument(
    "--people",
    default="data/processed/people.json",
    type=str,
    help="path to papers json"
)
parser.add_argument(
    "--max_seq_len",
    default=512,
    type=int,
    help="max seq len"
)
parser.add_argument(
    "--context_size",
    default=0,
    type=int,
    help="num messages to include in context"
)
parser.add_argument(
    "--working_dir",
    default='working_dir',
    type=str,
    help="num messages to include in context"
)
parser.add_argument(
    "--epochs",
    default=70,
    type=int,
    help="fine tuning epochs"
)
parser.add_argument(
    "--batch_size",
    default=10,
    type=int,
    help="fine tuning epochs"
)
parser.add_argument(
    "--learning_rate",
    default=2e-5,
    type=int,
    help="fine tuning epochs"
)
parser.add_argument(
    "--seed",
    default=420,
    type=int,
    help="fine tuning epochs"
)
parser.add_argument(
    "--length_discard",
    action='store_true',
    help="discard examples that are too long"
)
parser.add_argument(
    "--include_metadata",
    action='store_true',
    help="discard examples that are too long"
)
parser.add_argument(
    "--downsample",
    default=0.4,
    type=float,
    help="1-p where p = prop examples to throw out"
)
ARGS = parser.parse_args()


random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)




# this doesn't help
def get_time(dt_string):
    x = datetime.datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S.%f')
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



# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def get_raw_data(people, papers):
    sentences = []
    labels = []

    out = defaultdict(list)

    with open(people) as f:
        people = json.load(f)
    with open(papers) as f:
        papers = list(json.load(f).items())

    # pull out the raw sequences from data
    for pid, paper_info in papers:
        winners = paper_info["winning_reviewers"]
        losers = paper_info["losing_reviewers"]
        if paper_info["review_consults"] is None:
            continue
        for i, consult in enumerate(paper_info["review_consults"]):

            author = consult["author"]
            author_info = people.get(author, None)
            if author_info is None:
                continue
            if author not in winners + losers:
                continue
            ###################
            # (1) get raw inputs 
            context = paper_info["review_consults"][max(0, i - ARGS.context_size) : i]
            context = [c['text'] for c in context]
            text = consult["text"]
            label = int(author in winners)

            order = get_order(i, len(paper_info["review_consults"]))
            time = get_time(consult['date'])
            gender = people[consult['author']]['gender']
            try:
                university = people[consult['author']]['institution'].strip()
            except AttributeError:
                university = 'unknown'
            experience = str(round(math.log(len(people[consult['author']]['pids'])), 3))

            metadata = 'time: %s; gender: %s; university: %s; experience: %s order: %s' % (
                time, gender, university, experience, order)

            # downsample negatives
            if label == 0 and random.random() > ARGS.downsample:
                continue    

            ###################
            # (2) get input ids from inputs 
            sent = text + ' ' + metadata if ARGS.include_metadata else text
            encoded_sent = tokenizer.encode(sent, add_special_tokens=True)

            seg_id = 0
            segment_ids = [seg_id % 2] * len(encoded_sent)
            seg_id += 1

            # add context if needed
            if len(context) > 0 and len(encoded_sent) < ARGS.max_seq_len:
                encoded_sent = encoded_sent[1:] # strip [CLS]
                segment_ids = segment_ids[1:]

                for ci in context[::-1]:
                    encoded_ci = tokenizer.encode(ci, add_special_tokens=False)
                    encoded_sent = encoded_ci + [102] + encoded_sent # add [SEP]
                    segment_ids = ([seg_id % 2] * (len(encoded_ci) + 1)) + segment_ids
                    seg_id += 1

                # front-truncate if over here so that tgt sentence doesn't get chopped off. 
                # and add [CLS] back in
                encoded_sent = encoded_sent[-(ARGS.max_seq_len - 1):]
                segment_ids = segment_ids[-(ARGS.max_seq_len - 1):]

                encoded_sent = [101] + encoded_sent
                segment_ids = [segment_ids[0]] + segment_ids

            if ARGS.length_discard and len(encoded_sent) >= ARGS.max_seq_len:
                continue

            out['segment_ids'].append(segment_ids)
            out['input_ids'].append(encoded_sent)
            out['contexts'].append(context)
            out['sentences'].append(text)
            out['labels'].append(label)
            out['metadata'].append(metadata)

    # truncate + pad
    out['input_ids'] = pad_sequences(
        out['input_ids'], 
        maxlen=ARGS.max_seq_len, 
        dtype="long", 
        value=0, 
        truncating="post", 
        padding="post")

    out['segment_ids'] = pad_sequences(
        out['segment_ids'], 
        maxlen=ARGS.max_seq_len, 
        dtype="long", 
        value=0, 
        truncating="post", 
        padding="post")

    # get attn masks
    for sent in out['input_ids']:
        mask = [int(tok_id > 0) for tok_id in sent]
        out['attn_masks'].append(mask)

    return out


def build_dataloader(*args, sampler='random'):
    data = (torch.tensor(x) for x in args)
    data = TensorDataset(*data)

    sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=ARGS.batch_size)

    return dataloader


if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)
    os.makedirs(ARGS.working_dir + '/events')

writer = SummaryWriter(ARGS.working_dir + '/events')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


if os.path.exists(ARGS.working_dir + "/data.cache.pkl"):
    data = pickle.load(open(ARGS.working_dir + "/data.cache.pkl", 'rb'))
else:
    data = get_raw_data(ARGS.people, ARGS.papers)
    pickle.dump(data, open(ARGS.working_dir + "/data.cache.pkl", 'wb'))

train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks, train_segs, test_segs = train_test_split(
    data['input_ids'], data['labels'], data['attn_masks'], data['segment_ids'],
    random_state=ARGS.seed, test_size=0.1)

train_dataloader = build_dataloader(
    train_inputs, train_labels, train_masks, train_segs)
test_dataloader = build_dataloader(
    test_inputs, test_labels, test_masks, test_segs,
    sampler='order')

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=True,
    output_hidden_states=False,
)

optimizer = AdamW(model.parameters(), lr=ARGS.learning_rate, eps=1e-8)

total_steps = len(train_dataloader) * ARGS.epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps)

if CUDA:
    model = model.cuda()

for epoch_i in range(0, ARGS.epochs):
    
    # ========================================
    #               Training
    # ========================================
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, ARGS.epochs))
    print('Training...')

    losses = []
    t0 = time.time()
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. Loss: {:.2f}'.format(
                step, len(train_dataloader), elapsed, float(np.mean(losses))))

        if CUDA:
            batch = (x.cuda() for x in batch)            
        input_ids, labels, masks, segs = batch
        model.zero_grad()        

        outputs = model(
            input_ids,
            token_type_ids=None, 
            attention_mask=masks, 
            labels=labels)
        
        loss, _, _ = outputs
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_loss = np.mean(losses)
    writer.add_scalar('train/loss', np.mean(avg_loss), epoch_i)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    log = open(ARGS.working_dir + '/epoch%d.log' % epoch_i, 'w')
    for batch in test_dataloader:

        if CUDA:
            batch = (x.cuda() for x in batch)            
        input_ids, labels, masks, segs = batch

        with torch.no_grad():        
            outputs = model(
                input_ids,
                token_type_ids=segs, 
                attention_mask=masks, 
                labels=labels)
        loss, logits, attns = outputs

        losses.append(loss.item())

        labels = labels.cpu().numpy()
        input_ids = input_ids.cpu().numpy()
        preds = scipy.special.softmax(logits.cpu().numpy(), axis=1)
        input_toks = [
            tokenizer.convert_ids_to_tokens(s) for s in input_ids
        ]

        for seq, label, pred in zip(input_toks, labels, preds):
            sep_char = '+' if np.argmax(pred) == label else '-'
            log.write(sep_char * 40 + '\n')
            log.write(' '.join(seq) + '\n')
            log.write('label: ' + str(label) + '\n')
            log.write('pred: ' + str(np.argmax(pred)) + '\n')
            log.write('dist: ' + str(pred) + '\n')
            log.write('\n\n')

            all_preds += [pred]
            all_labels += [label]
    log.close()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = np.mean(losses)
    f1 = sklearn.metrics.f1_score(all_labels, np.argmax(all_preds, axis=1))
    acc = sklearn.metrics.accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    auc = sklearn.metrics.roc_auc_score(all_labels, all_preds[:, 1])

    writer.add_scalar('eval/acc', acc, epoch_i)
    writer.add_scalar('eval/auc', auc, epoch_i)
    writer.add_scalar('eval/f1', f1, epoch_i)
    writer.add_scalar('eval/loss', f1, epoch_i)

    print("  Loss: {0:.2f}".format(avg_loss))
    print("  Accuracy: {0:.2f}".format(acc))
    print("  F1: {0:.2f}".format(f1))
    print("  AUC: {0:.2f}".format(auc))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Done!")

