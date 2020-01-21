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
            label = int(author in winners)

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




class Bert(BertPreTrainedModel):
    def __init__(self, config, cls_num_labels=2):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)
        
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask)

        cls_logits = self.cls_classifier(pooled_output)
        cls_logits = self.cls_dropout(cls_logits)
        return cls_logits

def build_optimizer_scheduler(model, num_train_steps, learning_rate):
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, 
        num_training_steps=num_train_steps)

    return optimizer, scheduler



def build_loss_fn(debias_weight=None, num_labels=2):
    if CUDA:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    def cross_entropy_loss(logits, labels):
        return criterion(
            logits.view(-1, num_labels),
            labels.view(-1).type('torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))

    return cross_entropy_loss


def train_for_epoch(model, train_dataloader, loss_fn, optimizer, scheduler):
    losses = []
    for step, batch in enumerate(tqdm(train_dataloader)):
        # if step > 1: break
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        input_ids, input_masks, segment_ids, labels = batch
        loss, logits = model(
            input_ids, labels=labels)
        print('TRAIN')
        print(input_ids)
        print(loss)
        print(logits)
        print(labels)
        # loss = loss_fn(logits, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        losses.append(loss.detach().cpu().numpy())

    return losses

def run_inference(model, eval_dataloader, loss_fn, tokenizer, epoch=0):

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

    log = open(ARGS.working_dir + '/epoch%d.log' % epoch, 'w')

    all_labels = []
    all_preds = []
    all_losses = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        # if step > 1: break
        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        input_ids, input_masks, segment_ids, labels = batch
        with torch.no_grad():
            loss, logits = model(input_ids, labels=labels)
            # print(input_ids)
            # print(labels)\
            print('EVAL')
            print(input_ids)
            print(loss)
            print(labels)
            print(logits)
            print('#' * 100)
            # loss = loss_fn(logits, labels)
        input_toks = [
            tokenizer.convert_ids_to_tokens(seq) for seq in input_ids.cpu().numpy()
        ]
        labels = labels.cpu().numpy()
        preds = scipy.special.softmax(logits.detach().cpu().numpy(), axis=1)

        for seq, label, pred in zip(input_toks, labels, preds):
            sep_char = '+' if np.argmax(pred) == label else '-'
            log.write(sep_char * 40 + '\n')
            log.write('\n'.join(detokenize(seq).split('[SEP]')) + '\n')
            log.write('label: ' + str(label) + '\n')
            log.write('pred: ' + str(pred) + '\n')
            log.write('\n\n')

            all_preds += [pred]
            all_labels += [label]

        all_losses.append(float(loss.cpu().numpy()))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1 = sklearn.metrics.f1_score(all_labels, np.argmax(all_preds, axis=1))
    acc = sklearn.metrics.accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    auc = sklearn.metrics.roc_auc_score(all_labels, all_preds[:, 1])

    return acc, f1, auc

    return out


if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)
    os.makedirs(ARGS.working_dir + '/events')


writer = SummaryWriter(ARGS.working_dir + '/events')
# writer.add_scalar('test', 0, 0)
# writer.add_scalar('test', 0.5, 1)
# quit()

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
    cache_dir=ARGS.working_dir + '/cache')
if CUDA:
    model = model.cuda()


optimizer, scheduler = build_optimizer_scheduler(
    model, int((num_train_examples * ARGS.epochs) / ARGS.batch_size),
    ARGS.learning_rate)

loss_fn = build_loss_fn()

for epoch in range(ARGS.epochs):
    while True:
        print('TRAIN %d' % epoch)
        model.train()
        losses = train_for_epoch(model, test_iterator, loss_fn, optimizer, scheduler)
        # losses = [0]
        writer.add_scalar('train/loss', np.mean(losses), epoch + 1)
        print('EVAL')
        model.eval()
        acc, f1, auc = run_inference(model, test_iterator, loss_fn, tokenizer, epoch=epoch)
        writer.add_scalar('eval/acc', acc, epoch + 1)
        writer.add_scalar('eval/auc', auc, epoch + 1)
        writer.add_scalar('eval/f1', f1, epoch + 1)


writer.close()