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
import numpy as np
import time
import datetime
import random
from collections import defaultdict
import argparse
import os

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
    default=64,
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
    default=4,
    type=int,
    help="fine tuning epochs"
)
parser.add_argument(
    "--batch_size",
    default=32,
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
ARGS = parser.parse_args()


random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)



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
            # if author not in winners + losers:
            #     continue

            text = consult["text"]
            label = int(author in winners)

            if label == 0 and random.random() > 0.4:
                continue    


            out['sentences'].append(text)
            out['labels'].append(label)

    return out


def sentences2ids(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []

    for sent in sentences:
        # prepends cls and appends sep
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        input_ids.append(encoded_sent)
    input_ids = pad_sequences(input_ids, maxlen=ARGS.max_seq_len, dtype="long", 
                              value=0, truncating="post", padding="post")
    return input_ids

def ids2masks(input_ids):
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks



def build_dataloader(*args, sampler='random'):
    data = (torch.tensor(x) for x in args)
    data = TensorDataset(*data)

    sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=ARGS.batch_size)

    return dataloader


if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)
    os.makedirs(ARGS.working_dir + '/events')

if os.path.exists(ARGS.working_dir + "/data.cache.pkl"):
    data = pickle.load(open(ARGS.working_dir + "/data.cache.pkl", 'rb'))
    input_ids = pickle.load(open(ARGS.working_dir + "/ids.cache.pkl", 'rb'))
else:
    data = get_raw_data(ARGS.people, ARGS.papers)
    input_ids = sentences2ids(data['sentences'])

    pickle.dump(data, open(ARGS.working_dir + "/data.cache.pkl", 'wb'))
    pickle.dump(input_ids, open(ARGS.working_dir + "/ids.cache.pkl", 'wb'))

masks = ids2masks(input_ids)

train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
    input_ids, data['labels'], masks, 
    random_state=ARGS.seed, test_size=0.1)

train_dataloader = build_dataloader(
    train_inputs, train_labels, train_masks)
test_dataloader = build_dataloader(
    test_inputs, test_labels, test_masks, sampler='order')

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

losses = []
for epoch_i in range(0, ARGS.epochs):
    
    # ========================================
    #               Training
    # ========================================
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, ARGS.epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. Loss: {.2f}'.format(
                step, len(train_dataloader), elapsed, np.mean(losses)))
        input_ids, labels, masks = batch

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    losses.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        print(logits)
        print(label_ids)
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

