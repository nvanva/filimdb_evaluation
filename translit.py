import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import itertools as it
import collections as col
import random
import os
import copy
import json
from tqdm import tqdm
import datetime, time

from translit_utils.data import TextEncoder, load_datasets, create_dataloader
from translit_utils.metrics import compute_metrics


class Embedding(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Embedding, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.emb_layer(x)

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_size, requires_grad=False)
        # TODO: implement your code here 
        pe = pe.unsqueeze(0)
        # pe shape: (1, max_len, hidden_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: shape (batch size, sequence length, hidden size)
        x = x + self.pe[:, :x.size(1)]
        return x

class LayerNorm(nn.Module):
    "Layer Normalization layer"

    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.eps) + self.bias

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))

def padding_mask(x, pad_idx=0):
    assert len(x.size()) >= 2
    return (x != pad_idx).unsqueeze(-2)

def look_ahead_mask(size):
    "Mask out the right context"
    attn_shape = (1, size, size)
    look_ahead_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(look_ahead_mask) == 0

def compositional_mask(x, pad_idx=0):
    pm = padding_mask(x, pad_idx=pad_idx)
    seq_length = x.size(-1)
    result_mask = pm & \
                  look_ahead_mask(seq_length).type_as(pm.data)
    return result_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, dropout=None):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % n_heads == 0
        self.head_hidden_size = hidden_size // n_heads
        self.n_heads = n_heads
        self.linears = clone_layer(nn.Linear(hidden_size, hidden_size), 4)
        self.attn_weights = None
        self.dropout = dropout
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(p=self.dropout)

    def attention(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'
            query, key and value tensors have the same shape:
                (batch size, number of heads, sequence length, head hidden size)
            mask shape: (batch size, 1, sequence length, sequence length)
                '1' dimension value will be broadcasted to number of heads inside your operations
            mask should be applied before using softmax to get attn_weights
        """
        ## attn_weights shape: (batch size, number of heads, sequence length, sequence length)
        ## output shape: (batch size, number of heads, sequence length, head hidden size)
        ## TODO: provide your implementation here
        ## don't forget to apply dropout to attn_weights if self.dropout is not None
        raise NotImplementedError
        return output, attn_weights

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # Split vectors for different attention heads (from hidden_size => n_heads x head_hidden_size)
        # and do separate linear projection, for separate trainable weights
        query, key, value = \
            [l(x).view(batch_size, -1, self.n_heads, self.head_hidden_size).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn_weights = self.attention(query, key, value, mask=mask)
        # x shape: (batch size, number of heads, sequence length, head hidden size)
        # self.attn_weights shape: (batch size, number of heads, sequence length, sequence length)

        # Concatenate the output of each head
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.n_heads * self.head_hidden_size)

        return self.linears[-1](x)

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.pre_linear = nn.Linear(hidden_size, ff_hidden_size)
        self.post_linear = nn.Linear(ff_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.post_linear(self.dropout(F.relu(self.pre_linear(x))))

def clone_layer(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, hidden_size, ff_hidden_size, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, hidden_size,
                                            dropout=dropout['attention'])
        self.feed_forward = FeedForward(hidden_size, ff_hidden_size,
                                        dropout=dropout['relu'])
        self.sublayers = clone_layer(SublayerConnection(hidden_size, dropout['residual']), 2)

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedder = Embedding(config['hidden_size'],
                                  config['src_vocab_size'])
        self.positional_encoder = PositionalEncoding(config['hidden_size'],
                                                     max_len=config['max_src_seq_length'])
        self.embedding_dropout = nn.Dropout(p=config['dropout']['embedding'])
        self.encoder_layer = EncoderLayer(config['hidden_size'],
                                          config['ff_hidden_size'],
                                          config['n_heads'],
                                          config['dropout'])
        self.layers = clone_layer(self.encoder_layer, config['n_layers'])
        self.layer_norm = LayerNorm(config['hidden_size'])

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        x = self.embedding_dropout(self.positional_encoder(self.embedder(x)))
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)

class DecoderLayer(nn.Module):
    """
    Decoder is made of 3 sublayers: self attention, encoder-decoder attention
    and feed forward"
    """

    def __init__(self, hidden_size, ff_hidden_size, n_heads, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_heads, hidden_size,
                                            dropout=dropout['attention'])
        self.encdec_attn = MultiHeadAttention(n_heads, hidden_size,
                                              dropout=dropout['attention'])
        self.feed_forward = FeedForward(hidden_size, ff_hidden_size,
                                        dropout=dropout['relu'])
        self.sublayers = clone_layer(SublayerConnection(hidden_size, dropout['residual']), 3)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, decoder_mask))
        x = self.sublayers[1](x, lambda x: self.encdec_attn(x, encoder_output,
                                                            encoder_output, encoder_mask))
        return self.sublayers[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.embedder = Embedding(config['hidden_size'],
                                  config['tgt_vocab_size'])
        self.positional_encoder = PositionalEncoding(config['hidden_size'],
                                                     max_len=config['max_tgt_seq_length'])
        self.embedding_dropout = nn.Dropout(p=config['dropout']['embedding'])
        self.decoder_layer = DecoderLayer(config['hidden_size'],
                                          config['ff_hidden_size'],
                                          config['n_heads'],
                                          config['dropout'])
        self.layers = clone_layer(self.decoder_layer, config['n_layers'])
        self.layer_norm = LayerNorm(config['hidden_size'])

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.embedding_dropout(self.positional_encoder(self.embedder(x)))
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return self.layer_norm(x)

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.proj = nn.Linear(config['hidden_size'], config['tgt_vocab_size'])

        self.pad_idx = config['pad_idx']
        self.tgt_vocab_size = config['tgt_vocab_size']

    def encode(self, encoder_input, encoder_input_mask):
        return self.encoder(encoder_input, encoder_input_mask)

    def decode(self, encoder_output, encoder_input_mask, decoder_input, decoder_input_mask):
        return self.decoder(decoder_input, encoder_output, encoder_input_mask, decoder_input_mask)

    def linear_project(self, x):
        return self.proj(x)

    def forward(self, encoder_input, decoder_input):
        encoder_input_mask = padding_mask(encoder_input, pad_idx=self.config['pad_idx'])
        decoder_input_mask = compositional_mask(decoder_input, pad_idx=self.config['pad_idx'])
        encoder_output = self.encode(encoder_input, encoder_input_mask)
        decoder_output = self.decode(encoder_output, encoder_input_mask,
                                     decoder_input, decoder_input_mask)
        output_logits = self.linear_project(decoder_output)
        return output_logits


def prepare_model(config):
    model = Transformer(config)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class LrScheduler:
    def __init__(self, n_steps, **kwargs):
        self.type = kwargs['type']
        if self.type == 'warmup,decay_linear':
            ## TODO: provide your implementation here
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown type argument: {self.type}')
        self._step = 0
        self._lr = 0

    def step(self, optimizer):
        self._step += 1
        lr = self.learning_rate()
        for p in optimizer.param_groups:
            p['lr'] = lr

    def learning_rate(self, step=None):
        if step is None:
            step = self._step
        if self.type == 'warmup,decay_linear':
            ## TODO: provide your implementation here
            raise NotImplementedError
        return self._lr

    def state_dict(self):
        sd = copy.deepcopy(self.__dict__)
        return sd

    def load_state_dict(self, sd):
        for k in sd.keys():
            self.__setattr__(k, sd[k])


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def run_epoch(data_iter, model, lr_scheduler, optimizer, device, verbose=False):
    start = time.time()
    local_start = start
    total_tokens = 0
    total_loss = 0
    tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    for i, batch in tqdm(enumerate(data_iter)):
        encoder_input = batch[0].to(device)
        decoder_input = batch[1].to(device)
        decoder_target = batch[2].to(device)
        logits = model(encoder_input, decoder_input)
        loss = loss_fn(logits.view(-1, model.tgt_vocab_size),
                       decoder_target.view(-1))
        total_loss += loss.item()
        batch_n_tokens = (decoder_target != model.pad_idx).sum().item()
        total_tokens += batch_n_tokens
        if optimizer is not None:
            optimizer.zero_grad()
            lr_scheduler.step(optimizer)
            loss.backward()
            optimizer.step()

        tokens += batch_n_tokens
        if verbose and i % 1000 == 1:
            elapsed = time.time() - local_start
            print("batch number: %d, accumulated average loss: %f, tokens per second: %f" %
                  (i, total_loss / total_tokens, tokens / elapsed))
            local_start = time.time()
            tokens = 0

    average_loss = total_loss / total_tokens
    print('** End of epoch, accumulated average loss = %f **' % average_loss)
    epoch_elapsed_time = format_time(time.time() - start)
    print(f'** Elapsed time: {epoch_elapsed_time}**')
    return average_loss


def save_checkpoint(epoch, model, lr_scheduler, optimizer, model_dir_path):
    save_path = os.path.join(model_dir_path, f'cpkt_{epoch}_epoch')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict()
    }, save_path)
    print(f'Saved checkpoint to {save_path}')

def load_model(epoch, model_dir_path):
    save_path = os.path.join(model_dir_path, f'cpkt_{epoch}_epoch')
    checkpoint = torch.load(save_path)
    with open(os.path.join(model_dir_path, 'model_config.json'), 'r', encoding='utf-8') as rf:
        model_config = json.load(rf)
    model = prepare_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def greedy_decode(model, device, encoder_input, max_len, start_symbol):
    batch_size = encoder_input.size()[0]
    decoder_input = torch.ones(batch_size, 1).fill_(start_symbol).type_as(encoder_input.data).to(device)

    for i in range(max_len):
        logits = model(encoder_input, decoder_input)

        _, predicted_ids = torch.max(logits, dim=-1)
        next_word = predicted_ids[:, i]
        # print(next_word)
        rest = torch.ones(batch_size, 1).type_as(decoder_input.data)
        # print(rest[:,0].size(), next_word.size())
        rest[:, 0] = next_word
        decoder_input = torch.cat([decoder_input, rest], dim=1).to(device)
        # print(decoder_input)
    return decoder_input

def generate_predictions(dataloader, max_decoding_len, text_encoder, model, device):
    # print(f'Max decoding length = {max_decoding_len}')
    model.eval()
    predictions = []
    start_token_id = text_encoder.service_vocabs['token2id'][
        text_encoder.service_token_names['start_token']]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            encoder_input = batch[0].to(device)
            prediction_tensor = \
                greedy_decode(model, device, encoder_input, max_decoding_len,
                              start_token_id)

            predictions.extend([''.join(e) for e in text_encoder.id2token(prediction_tensor.cpu().numpy(),
                                                                          unframe=True, lang_key='ru')])
    return np.array(predictions)



def train(source_strings, target_strings):
    '''Common training cycle for final run (fixed hyperparameters,
    no evaluation during training)'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU device: {device}')
    else:
        device = torch.device('cpu')
        print(f'GPU is not available, using CPU device {device}')

    train_df = pd.DataFrame({'en': source_strings, 'ru': target_strings})
    text_encoder = TextEncoder()
    text_encoder.make_vocabs(train_df)

    model_config = {
        'src_vocab_size': text_encoder.src_vocab_size,
        'tgt_vocab_size': text_encoder.tgt_vocab_size,
        'max_src_seq_length': max(train_df['en'].aggregate(len)) + 2, #including start_token and end_token
        'max_tgt_seq_length': max(train_df['ru'].aggregate(len)) + 2,
        'n_layers': 2,
        'n_heads': 2,
        'hidden_size': 128,
        'ff_hidden_size': 256,
        'dropout': {
            'embedding': 0.1,
            'attention': 0.1,
            'residual': 0.1,
            'relu': 0.1
        },
        'pad_idx': 0
    }
    # model = load_model(epoch=,model_dir_path=)
    # model.to(device)
    model = prepare_model(model_config)
    model.to(device)

    train_config = {'batch_size': 200, 'n_epochs': 600, 'lr_scheduler': {
        'type': 'warmup,decay_linear',
        'warmup_steps_part': 0.1,
        'lr_peak': 3e-4,
    }}

    #Model training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=0.)
    n_steps = (len(train_df) // train_config['batch_size'] + 1) * train_config['n_epochs']
    lr_scheduler = LrScheduler(n_steps, **train_config['lr_scheduler'])

    # prepare train data
    source_strings, target_strings = zip(*sorted(zip(source_strings, target_strings),
                                                 key=lambda e: len(e[0])))
    train_dataloader = create_dataloader(source_strings, target_strings, text_encoder,
                                         train_config['batch_size'],
                                         shuffle_batches_each_epoch=True)
    # training cycle
    for epoch in range(1,train_config['n_epochs']+1):
        print('\n' + '-'*40)
        print(f'Epoch: {epoch}')
        print(f'Run training...')
        model.train()
        run_epoch(train_dataloader, model,
                  lr_scheduler, optimizer, device=device, verbose=False)
    learnable_params = {
        'model': model,
        'text_encoder': text_encoder,
    }
    return learnable_params

def classify(source_strings, learnable_params):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU device: {device}')
    else:
        device = torch.device('cpu')
        print(f'GPU is not available, using CPU device {device}')

    model = learnable_params['model']
    text_encoder = learnable_params['text_encoder']
    batch_size = 200
    dataloader = create_dataloader(source_strings, None, text_encoder,
                                   batch_size, shuffle_batches_each_epoch=False)
    max_decoding_len = model.config['max_tgt_seq_length']
    predictions = generate_predictions(dataloader, max_decoding_len, text_encoder, model, device)
    #return single top1 prediction for each sample
    return np.expand_dims(predictions, 1)

if __name__=='__main__':
    pass
    # seed_val = 42
    # random.seed(seed_val)
    # np.random.seed(seed_val)
    # torch.manual_seed(seed_val)
    # torch.cuda.manual_seed_all(seed_val)
    # data_dir_path = './data'
    # parts = ['train', 'test']
    # datasets = load_datasets(data_dir_path, parts)
    # train_source_strings = datasets['train']['en']
    # train_target_strings = datasets['train']['ru']
    # learnable_params = train(train_source_strings, train_target_strings)
    #
    # test_source_strings = datasets['test']['en']
    # test_target_strings = datasets['test']['ru']
    # preds = classify(test_source_strings, learnable_params)
    # mv = compute_metrics(np.squeeze(preds), test_target_strings, ['acc@1', 'mean_ld@1'])
    # print(mv)
