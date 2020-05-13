import copy
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import itertools as it
import collections as col
import random

def load_datasets(data_dir_path, parts):
    datasets = {}
    for part in parts:
        path = os.path.join(data_dir_path, f'{part}.tsv')
        datasets[part] = pd.read_csv(path, sep='\t', na_filter=False)
        print(f'Loaded {part} dataset, length: {len(datasets[part])}')
    return datasets

class TextEncoder:
    def __init__(self, load_dir_path=None):
        self.lang_keys = ['en', 'ru']
        self.directions = ['id2token', 'token2id']
        self.service_token_names = {
            'pad_token': '<pad>',
            'start_token': '<start>',
            'unk_token': '<unk>',
            'end_token': '<end>'
        }
        service_id2token = dict(enumerate(self.service_token_names.values()))
        service_token2id ={v:k for k,v in service_id2token.items()}
        self.service_vocabs = dict(zip(self.directions,
                                       [service_id2token, service_token2id]))
        if load_dir_path is None:
            self.vocabs = {}
            for lk in self.lang_keys:
                self.vocabs[lk] = copy.deepcopy(self.service_vocabs)
        else:
            self.vocabs = self.load_vocabs(load_dir_path)
    def load_vocabs(self, load_dir_path):
        vocabs = {}
        load_path = os.path.join(load_dir_path, 'vocabs')
        for lk in self.lang_keys:
            vocabs[lk] = {}
            for d in self.directions:
                columns = d.split('2')
                print(lk, d)
                df = pd.read_csv(os.path.join(load_path, f'{lk}_{d}'))
                vocabs[lk][d] = dict(zip(*[df[c] for c in columns]))
        return vocabs
    
    def save_vocabs(self, save_dir_path):
        save_path = os.path.join(save_dir_path, 'vocabs')
        os.makedirs(save_path, exist_ok=True)
        for lk in self.lang_keys:
            for d in self.directions:
                columns = d.split('2')
                pd.DataFrame(data=self.vocabs[lk][d].items(),
                    columns=columns).to_csv(os.path.join(save_path, f'{lk}_{d}'),
                                                index=False,
                                                sep=',')
    def make_vocabs(self, data_df):
        for lk in self.lang_keys:
            tokens = col.Counter(''.join(list(it.chain(*data_df[lk])))).keys()
            part_id2t = dict(enumerate(tokens, start=len(self.service_token_names)))
            part_t2id = {k:v for v,k in part_id2t.items()}
            part_vocabs = [part_id2t, part_t2id]
            for i in range(len(self.directions)):
                self.vocabs[lk][self.directions[i]].update(part_vocabs[i])
                
        self.src_vocab_size = len(self.vocabs['en']['id2token'])
        self.tgt_vocab_size = len(self.vocabs['ru']['id2token'])
                
    def frame(self, sample, start_token=None, end_token=None):
        if start_token is None:
            start_token=self.service_token_names['start_token']
        if end_token is None:
            end_token=self.service_token_names['end_token']
        return [start_token] + sample + [end_token]
    def token2id(self, samples, frame, lang_key):
        if frame:
            samples = list(map(self.frame, samples))
        vocab = self.vocabs[lang_key]['token2id']
        return list(map(lambda s:
                        [vocab[t] if t in vocab.keys() else vocab[self.service_token_names['unk_token']]
                         for t in s], samples))
    
    def unframe(self, sample, start_token=None, end_token=None):
        if start_token is None:
            start_token=self.service_vocabs['token2id'][self.service_token_names['start_token']]
        if end_token is None:
            end_token=self.service_vocabs['token2id'][self.service_token_names['end_token']]
        pad_token=self.service_vocabs['token2id'][self.service_token_names['pad_token']]
        return list(it.takewhile(lambda e: e != end_token and e != pad_token, sample[1:]))
    def id2token(self, samples, unframe, lang_key):
        if unframe:
            samples = list(map(self.unframe, samples))
        vocab = self.vocabs[lang_key]['id2token']
        return list(map(lambda s:
                        [vocab[idx] if idx in vocab.keys() else self.service_token_names['unk_token'] for idx in s], samples))

class TranslitData(torch_data.Dataset):
    def __init__(self, source_strings, target_strings,
                text_encoder):
        super(TranslitData, self).__init__()
        self.source_strings = source_strings
        self.text_encoder = text_encoder
        if target_strings is not None:
            assert len(source_strings) == len(target_strings)
            self.target_strings = target_strings
        else:
            self.target_strings = None
    def __len__(self):
        return len(self.source_strings)
    def __getitem__(self, idx):
        src_str = self.source_strings[idx]
        encoder_input = self.text_encoder.token2id([list(src_str)], frame=True, lang_key='en')[0]
        if self.target_strings is not None:
            tgt_str = self.target_strings[idx]
            tmp = self.text_encoder.token2id([list(tgt_str)], frame=True, lang_key='ru')[0]
            decoder_input = tmp[:-1]
            decoder_target = tmp[1:]
            return (encoder_input, decoder_input, decoder_target)
        else:
            return (encoder_input,)

class BatchSampler(torch_data.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, shuffle_each_epoch):
        super(BatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.batches = []
        for b in super(BatchSampler, self).__iter__():
            self.batches.append(b)
        self.shuffle_each_epoch = shuffle_each_epoch
        if self.shuffle_each_epoch:
            random.shuffle(self.batches)
        self.index = 0
        #print(f'Batches collected: {len(self.batches)}')
    def __iter__(self):
        self.index = 0
        return self
    def __next__(self):
        if self.index == len(self.batches):
            if self.shuffle_each_epoch:
                random.shuffle(self.batches)
            raise StopIteration
        else:
            batch = self.batches[self.index]
            self.index += 1
            return batch

def collate_fn(batch_list):
    '''batch_list can store either 3 components:
        encoder_inputs, decoder_inputs, decoder_targets
        or single component: encoder_inputs'''
    components = list(zip(*batch_list))
    batch_tensors = []
    for data in components:
        max_len = max([len(sample) for sample in data])
        #print(f'Maximum length in batch = {max_len}')
        sample_tensors = [torch.tensor(s, requires_grad=False, dtype=torch.int64)
                         for s in data]
        batch_tensors.append(nn.utils.rnn.pad_sequence(
            sample_tensors,
            batch_first=True, padding_value=0))
    return tuple(batch_tensors) 

def create_dataloader(source_strings, target_strings,
                      text_encoder, batch_size,
                      shuffle_batches_each_epoch):
    '''target_strings parameter can be None'''
    dataset = TranslitData(source_strings, target_strings,
                                text_encoder=text_encoder)
    seq_sampler = torch_data.SequentialSampler(dataset)
    batch_sampler = BatchSampler(seq_sampler, batch_size=batch_size,
                                drop_last=False,
                                shuffle_each_epoch=shuffle_batches_each_epoch)
    dataloader = torch_data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       collate_fn=collate_fn)
    return dataloader
