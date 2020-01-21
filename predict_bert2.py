"""
try and predict which comments belong to winners with bert

"""
import argparse
from transformers import BertTokenizer
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertSelfAttention, BertForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import os
import json
from tqdm import tqdm
from collections import defaultdict
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from tensorboardX import SummaryWriter
import scipy
import sklearn
import numpy as np
import random
import torch

random.seed(420)

CUDA = (torch.cuda.device_count() > 0)

max_grad_norm = 1.0

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
    default=3,
    type=int,
    help="fine tuning epochs"
)
parser.add_argument(
    "--batch_size",
    default=8,
    type=int,
    help="fine tuning epochs"
)
parser.add_argument(
    "--learning_rate",
    default=1e-5,
    type=int,
    help="fine tuning epochs"
)
ARGS = parser.parse_args()



def get_examples(people, papers, tokenizer, max_seq_len):
    cls_tok = '[CLS]'
    sep_tok = '[SEP]'
    pad_tok = '[PAD]'
    pad_idx = 0 # 0 in bert vocab

    def pad(id_arr, pad_val):
        return id_arr + ([pad_val] * (max_seq_len - len(id_arr)))

    def prep_sentance(s):
        toks = tokenizer.tokenize(s)

    with open(people) as f:
        people = json.load(f)

    out_examples = defaultdict(list)

    for pid, paper_info in tqdm(papers):
        winners = paper_info["winning_reviewers"]
        losers = paper_info["losing_reviewers"]
        if paper_info["review_consults"] is None:
            continue
        for i, consult in enumerate(paper_info["review_consults"]):

            author = consult["author"]
            author_info = people.get(author, None)
            if author_info is None:
                continue
            # if author not in winners + losers:
            #     continue

            context = paper_info["review_consults"][max(0, i - ARGS.context_size) : i]
            context = [c['text'] for c in context]
            text = consult["text"]

            input_toks = tokenizer.tokenize(text) + [sep_tok]
            segment_ids = [1] * len(input_toks)

            # Keep adding context
            i = 0
            while len(input_toks) < max_seq_len and len(context) > 0:
                new_toks = tokenizer.tokenize(context[-1]) + [sep_tok]
                segment_ids = ([i % 2] * len(new_toks)) + segment_ids
                input_toks = new_toks + input_toks
                i += 1
                context.pop()

            # front-truncate
            input_toks = input_toks[len(input_toks) - (max_seq_len - 1) :]
            segment_ids = segment_ids[len(segment_ids) - (max_seq_len - 1):]

            # add cls
            input_toks = [cls_tok] + input_toks
            segment_ids =  [segment_ids[0]] + segment_ids


            label = int(author in winners) #1 if len(input_toks) > 150 else 0 #


            # build mask
            input_mask = ([1] * len(input_toks)) + ([0] * (max_seq_len - len(input_toks)))

            # back-pad
            input_toks = pad(input_toks, pad_tok)
            segment_ids = pad(segment_ids, 0)


            input_ids = tokenizer.convert_tokens_to_ids(input_toks)

            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            out_examples['input_ids'].append(input_ids)
            out_examples['input_masks'].append(input_mask)
            out_examples['segment_ids'].append(segment_ids)
            out_examples['labels'].append(label)

    return out_examples


def get_dataloader(
    people_path, papers, tokenizer, batch_size, 
    cache=None, test=False):

    def collate(data):
        # group by datatype
        return [torch.stack(x) for x in zip(*data)]

    if cache is not None and os.path.exists(cache):
        examples = pickle.load(open(cache, 'rb'))
    else: 
        examples = get_examples(
            people_path, papers, 
            tokenizer, ARGS.max_seq_len)
        pickle.dump(examples, open(cache, 'wb'))

    data = TensorDataset(
        torch.tensor(examples['input_ids'], dtype=torch.long),
        torch.tensor(examples['input_masks'], dtype=torch.long),
        torch.tensor(examples['segment_ids'], dtype=torch.long),
        torch.tensor(examples['labels'], dtype=torch.long))

    dataloader = DataLoader(
        data, 
        sampler=(SequentialSampler(data) if test else RandomSampler(data)),
        collate_fn=collate,
        batch_size=ARGS.batch_size)

    return dataloader, len(examples['input_ids'])



def detokenize(s):
    out = ''
    for x in s:
        if x == '[PAD]':
            break
        if x.startswith('##'):
            out += x[2:]
        else:
            out += ' ' + x
    return out




if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)
    os.makedirs(ARGS.working_dir + '/events')


writer = SummaryWriter(ARGS.working_dir + '/events')

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', cache_dir=ARGS.working_dir + '/cache')

with open(ARGS.papers) as f:
    papers = json.load(f)
papers = list(papers.items())
random.shuffle(papers)
train_papers = papers[:-200]
test_papers = papers[-200:]

print("BUILDING TRAIN DATA...")
train_iterator, num_train_examples = get_dataloader(
    ARGS.people, train_papers, tokenizer, ARGS.batch_size, 
    ARGS.working_dir + '/data_cache.train.pkl', test=False)
print('DONE. %d EXAMPLES' % num_train_examples)

print("BUILDING TEST DATA...")
test_iterator, num_test_examples = get_dataloader(
    ARGS.people, test_papers, tokenizer, ARGS.batch_size, 
    ARGS.working_dir + '/data_cache.test.pkl', 
    test=True)
print('DONE. %d EXAMPLES' % num_test_examples)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    cache_dir=ARGS.working_dir + '/cache',
    output_attentions=True)
if CUDA:
    model = model.cuda()


# Parameters:
lr = 1e-7
max_grad_norm = 1.0
num_training_steps = num_train_examples * ARGS.epochs / ARGS.batch_size
warmup_proportion = 0.1
num_warmup_steps = num_training_steps * warmup_proportion

optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_training_steps)  # PyTorch scheduler


for epoch in range(ARGS.epochs):
    log = open(ARGS.working_dir + '/epoch%d.log' % epoch, 'w')

    losses = []
    for i, batch in enumerate(tqdm(train_iterator)):
        # continue
        if i > 100: continue
        input_ids, input_masks, segment_ids, labels = batch
        if CUDA:
            input_ids = input_ids.cuda()
            labels = labels.cuda().unsqueeze(0)
        outputs = model(input_ids, labels=labels)
        loss, logits, attentions = outputs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.cpu().detach().numpy()))

    writer.add_scalar('train/loss', np.mean(losses), epoch + 1)

    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    for i, batch in enumerate(tqdm(test_iterator)):
        input_ids, input_masks, segment_ids, labels = batch
        if CUDA:
            input_ids = input_ids.cuda()
            labels = labels.cuda().unsqueeze(0)
        outputs = model(input_ids, labels=labels)
        loss, logits, attentions = outputs

        losses.append(float(loss.cpu().detach().numpy()))

        preds = scipy.special.softmax(logits.detach().cpu().numpy(), axis=1)
        labels = np.squeeze(labels.cpu().numpy()).tolist()
        if not isinstance(labels, list): # singletons
            labels = [labels]
        input_toks = [
            tokenizer.convert_ids_to_tokens(seq) for seq in input_ids.cpu().numpy()
        ]
        for seq, label, pred in zip(input_toks, labels, preds):
            sep_char = '+' if np.argmax(pred) == label else '-'
            log.write(sep_char * 40 + '\n')
            log.write('\n'.join(detokenize(seq).split('[SEP]')) + '\n')
            log.write('label: ' + str(label) + '\n')
            log.write('pred: ' + str(pred) + '\n')
            log.write('\n\n')

            all_preds += [pred]
            all_labels += [label]

    writer.add_scalar('test/loss', np.mean(losses), epoch + 1)

    f1 = sklearn.metrics.f1_score(all_labels, np.argmax(all_preds, axis=1))
    acc = sklearn.metrics.accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    auc = sklearn.metrics.roc_auc_score(all_labels, np.array(all_preds)[:, 1])
    writer.add_scalar('eval/acc', acc, epoch + 1)
    writer.add_scalar('eval/auc', auc, epoch + 1)
    writer.add_scalar('eval/f1', f1, epoch + 1)

    log.close()
    # print('TRAIN %d' % epoch)
    # model.train()
    # losses = train_for_epoch(model, test_iterator, loss_fn, optimizer, scheduler)
    # # losses = [0]
    # writer.add_scalar('train/loss', np.mean(losses), epoch + 1)
    # print('EVAL')
    # model.eval()
    # acc, f1, auc = run_inference(model, test_iterator, loss_fn, tokenizer, epoch=epoch)
    # writer.add_scalar('eval/acc', acc, epoch + 1)
    # writer.add_scalar('eval/auc', auc, epoch + 1)
    # writer.add_scalar('eval/f1', f1, epoch + 1)


writer.close()