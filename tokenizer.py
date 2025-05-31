import torch
import torch.nn as nn
import regex as re
import json
import mmap
from tqdm import tqdm
from functools import lru_cache
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


# Byte Level BytePairEncoding
def bytes_to_unicode() -> dict[int, str]:
    '''
    Returns a mapping from bytes to unicode character
    '''
    # Characters that when printed, aren't meaningful -- to be shifted to a more meaningful repr
    to_shift = set(range(0, 33)) | set(range(127, 161)) | {173}

    # Keys of the dictionary -- each byte
    byte = range(0, 256)
    # Values of the dictionary -- each unicode char
    chars = []
    n = 0
    for i in byte:
        if i in to_shift:
            chars.append(chr(256 + n))
            n += 1
        else:
            chars.append(chr(i))
    return dict(zip(byte, chars))

class BPETokenizer(nn.Module):

    def __init__(self, vocab_path: str, merges_path: str):
        super().__init__()
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            merges = f.read()

        # Store merges as a list of tuples, remove last blank line
        merges = [tuple(merge_str.split()) for merge_str in merges.split("\n")[:-1]]

        # Token to/from BPE index mappings
        self.encoder = vocab
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Byte to/from unicode character mappings
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        # GPT-2 pre-tokenization splitting regex pattern
        self.pat = re.compile(rb"""
                                 's|'t|'re|'ve|'m|'ll|'d|  # Common contractions
                                 \ ?\p{L}+|\ ?\p{N}+|  # Optional space followed by 1+ unicode letter or number
                                 \ ?[^\s\p{L}\p{N}]+|  # Optional space followed by 1+ non-whitespace/letter/number
                                 \s+(?!\S)|  # 1+ whitespace characters not followed by non-whitespace
                                 \s+  # 1+ whitespace characters
                                 """, re.X)

    def forward(self, text, return_tensors=True, n_jobs=1):
        if isinstance(text, list):
            tokens = self.encode_batch(text, n_jobs)
        else:
            tokens = self.encode(text)
            tokens = torch.tensor(tokens) if return_tensors else tokens
        return tokens

    @lru_cache(maxsize=16384)
    def bpe(self, token):
        '''
        Applies merge rules on token
        '''
        chars = [char for char in token]
        # For each merge rule, attempt to merge any adjacent pairs of characters
        for pair in self.bpe_ranks.keys():
            i = 0
            while i < len(chars) - 1:
                if chars[i] == pair[0] and chars[i+1] == pair[1]:
                    chars = chars[:i] + ["".join(pair)] + chars[i+2:]
                else:
                    i += 1
        return chars

    def encode(self, text: str|bytes|mmap.mmap) -> list[int]:
        '''
        Encodes a string into BPE tokens
        '''
        bpe_tokens = []
        # Convert string to bytestring
        if (type(text) == str):
            text = text.encode("utf-8")

        # Splits text using the regex pattern to be fed into the BPE algorithm
        for token in tqdm(re.finditer(self.pat, text), desc="Tokenizing text"):
            # Transform token into its bytes representation, map the bytes to its unicode repr
            token = "".join(self.byte_encoder[b] for b in token[0])
            # Perform bpe merges on the token, then map results to their BPE indices according to the encoder
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token))
        return bpe_tokens

    def encode_batch(self, batch: list[str], n_jobs: int = 1) -> list[list[int]]:
        '''
        Encodes lists of strings into corresponding lists of BPE tokens
        '''
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            result = executor.map(self.encode, batch)
        return list(result)

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens, int):  # single token passed in
            tokens = [tokens]
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
        return text
    
    @property
    def vocab_len(self) -> int:
        return len(self.encoder)
    
    @staticmethod
    def merge(split_word, merge_rule):
        '''Apply merge rules on a split word'''
        i = 0
        new_token = merge_rule[0] + merge_rule[1]
        new_word = []
        # For each bigram in the word, attempt a merge
        while i < len(split_word) - 1:
            if (split_word[i], split_word[i+1]) == merge_rule:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(split_word[i])
                i += 1
        if i == len(split_word) - 1:
            new_word.append(split_word[i])
        return new_word
    
    @staticmethod
    def train_tokenizer(data, vocab_size, vocab_outfile=None, merges_outfile=None):
        if vocab_size < 256:
            raise ValueError("vocab_size must be greater than 256")

        # Pretokenize the data
        byte_encoder = bytes_to_unicode()
        pat_str = re.compile(rb"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        split_words = [
            [byte_encoder[b] for b in token[0]] for token in re.finditer(pat_str, data)
        ]
        # Add base vocabulary to the vocab
        vocab = set(byte_encoder.values())
        merges = []

        # Build the vocab until the desired vocab size is met
        with tqdm(total=vocab_size, initial=len(vocab), desc="Building vocabulary") as pbar:
            while len(vocab) < vocab_size:
                pair_freq = Counter()
                # Find the most common pair
                for split_word in split_words:
                    pair_freq.update(zip(split_word[:-1], split_word[1:]))
                most_common_pair = pair_freq.most_common(1)[0][0]

                # Update vocab and merges list
                new_token = most_common_pair[0] + most_common_pair[1]
                vocab.add(new_token)
                merges.append(most_common_pair)

                # Perform the merge on the data
                new_split_words = []
                while len(split_words) > 0:
                    split_word = split_words.pop()
                    i = 0
                    new_word = []
                    # For each bigram in the word, attempt a merge
                    while i < len(split_word) - 1:
                        if (split_word[i], split_word[i+1]) == most_common_pair:
                            new_word.append(new_token)
                            i += 2
                        else:
                            new_word.append(split_word[i])
                            i += 1
                    if i == len(split_word) - 1:
                        new_word.append(split_word[i])
                    new_split_words.append(new_word)
                split_words = new_split_words
                pbar.update(1)

        vocab = sorted(list(vocab))
        # Write to file
        if merges_outfile != None:
            with open(merges_outfile, "w", encoding="utf-8") as f:
                for merge in merges:
                    f.write(merge[0] + " " + merge[1] + "\n")
        if vocab_outfile != None:
            with open(vocab_outfile, "w", encoding="utf-8") as f:
                json.dump({v: i for i, v in enumerate(vocab)}, f, ensure_ascii=False)
        return vocab, merges
