"""
BPE (Byte Pair Encoding) training implementation.

This module implements the BPE algorithm for learning a tokenizer vocabulary
from a text corpus, compatible with GPT-2 style tokenization.
"""

from __future__ import annotations

import regex as re
from collections import Counter
from pathlib import Path
from typing import Iterator


# GPT-2 pre-tokenization pattern
# GPT的正则表达式，用来将一长串文本“切”成一个个小的片段（Tokens），直接按空格切分（Whitespace Split）会遇到很多问题
# 's|'t|'re|'ve|'m|'ll|'d：缩写词尾（如 I'm 中的 'm，don't 中的 't）
# ?\p{L}+ 匹配字母。前面的  ? 表示它会尽可能带上前面的一个空格
# ?\p{N}+ 匹配数字（如 123, 456）
# ?[^\s\p{L}\p{N}]+ 匹配标点符号或其他非字母数字字符。这确保了感叹号、问号等不会和单词粘连。
# ^：当它出现在方括号的开头时，代表取反（Not）。意思是“除了后面这些，其他的都要”。
# \s+(?!\S) 匹配结尾的空格。
# (?!\S)：这是一个负向先行断言（Negative Lookahead）。\S（大写 S）：代表非空白符（即任何看得见的文字、数字或标点）。(?!)：意思是“后面不能跟着...”。
# \s+ 匹配连续的空格
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE
)


def get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    """Get all adjacent pairs in a word (tuple of byte tokens)."""
    # 把一个单词的相邻字母进行配对：比如word = ('h', 'e', 'l', 'l', 'o')
    # 那么 pairs 就是 {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')}
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all occurrences of a pair in a word."""
    # 把一个input的word中出现在pair中的相邻字母进行合并：比如word = ('h', 'e', 'l', 'l', 'o')，pair = ('l', 'l')
    # 那么 new_word 就是 ('h', 'e', 'll', 'o')
    first, second = pair
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> Iterator[str]:
    """
    Pre-tokenize text using GPT-2 pattern, preserving special tokens.
    
    Special tokens are yielded as-is (not split by the regex pattern).
    """
    special_tokens = special_tokens or []
    
    if not special_tokens:
        # No special tokens, just use the pattern
        for match in GPT2_PAT.finditer(text):
            yield match.group() # 按照 GPT-2 的正则规则，把这段文本里所有符合条件的碎片，按顺序一个一个地找出来给我。
        return
    
    # Sort special tokens by length (longest first) for greedy matching
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    
    # Build a pattern that matches special tokens
    # 见tutorial
    # 这一部分的意思就是把特殊字符切出来
    import re as std_re
    special_pattern = "|".join(std_re.escape(s) for s in sorted_specials)
    split_pattern = f"({special_pattern})"
    
    # Split text by special tokens
    parts = std_re.split(split_pattern, text)
    # part 主要就是一些普通文本，或者特殊token。比如输入 "Hello!! <|endoftext|> !!world"，
    # special_tokens=["<|endoftext|>"]，那么 parts 就是 ["Hello!! ", "<|endoftext|>", " !!world"]
    # 总之就是既包含了普通文本，也包含了特殊token

    for part in parts:
        if part in special_tokens:
            # Special token - yield as-is, but it won't be BPE-encoded
            # (we skip special tokens in the word frequency counting)
            continue
        elif part:
            # Regular text - apply GPT-2 pre-tokenization 切出来
            for match in GPT2_PAT.finditer(part): 
                #GPT2_PAT 相当于就是把一个文本切成tokens, 
                # 比如说上面的 "Hello!! " 就会被切成 ["Hello", "!!", " "]
                yield match.group()


def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from a text file.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include (e.g., ["<|endoftext|>"])
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: dict mapping token_id (int) -> token (bytes)
        - merges: list of merge pairs in order they were learned [(bytes, bytes), ...]
    
    Algorithm Overview:
        BPE iteratively merges the most frequent pair of adjacent tokens until
        the vocabulary reaches the target size.
    
    Detailed Steps:
    
    1. VOCABULARY INITIALIZATION
       The initial vocabulary is built in this exact order:
       - First: Add special tokens (in the order provided)
       - Then: Add all 256 single-byte values (0x00 to 0xFF)
       
       Example with special_tokens=["<|endoftext|>"]:
         vocab = {
             0: b"<|endoftext|>",   # Special token first
             1: b"\\x00",           # Byte 0
             2: b"\\x01",           # Byte 1
             ...
             256: b"\\xff",         # Byte 255
         }
       
       So the initial vocab size = len(special_tokens) + 256
    
    2. WORD FREQUENCY COUNTING
       - Pre-tokenize the corpus using pre_tokenize(text, special_tokens)
       - For each pre-token, convert to bytes and represent as tuple of single bytes
       - Skip any word containing a "forbidden substring" (prefix of a special token)
       
       Example: "hello" -> (b'h', b'e', b'l', b'l', b'o')
       
       word_freqs is a Counter mapping: tuple[bytes, ...] -> frequency
    
    3. PAIR FREQUENCY COUNTING  
       Count how often each adjacent pair appears across ALL words, weighted by
       word frequency.
       
       Example: If word (b'h', b'e', b'l', b'l', b'o') appears 10 times:
         - pair (b'h', b'e') gets +10
         - pair (b'e', b'l') gets +10
         - pair (b'l', b'l') gets +10
         - pair (b'l', b'o') gets +10
    
    4. MERGE LOOP (repeat until vocab_size is reached)
       
       a. SELECT BEST PAIR (DETERMINISTIC TIE-BREAKING):
          Find the pair with highest frequency. If multiple pairs have the same
          frequency, select the lexicographically smallest pair.
          
          Lexicographic comparison on (bytes, bytes) tuples:
            - Compare first element as bytes
            - If equal, compare second element as bytes
          
          Example: If pairs (b'a', b'b') and (b'a', b'c') both have freq=100,
                   select (b'a', b'b') because b'b' < b'c'
          
          Implementation: max(pair_counts, key=lambda p: (pair_counts[p], p))
                          This sorts by (frequency, pair) and takes the max.
                          Since we want highest freq but lowest pair for ties,
                          use: max(pair_counts, key=lambda p: (pair_counts[p], p))
                          
                          Note: Python compares bytes lexicographically by default.
       
       b. CREATE MERGED TOKEN:
          new_token = first + second  (bytes concatenation)
          Add to vocabulary with next available token_id
          Append (first, second) to merges list
       
       c. UPDATE WORD REPRESENTATIONS:
          For each word in word_freqs, apply the merge using merge_word()
          This replaces all occurrences of the pair with the merged token
       
       d. UPDATE PAIR COUNTS:
          Recompute pair frequencies for the updated words
          (Or incrementally update - subtract old pairs, add new pairs)
    
    5. RETURN
       Return (vocab, merges) where merges is the list of pairs in the order
       they were merged.
    
    Performance Note:
        A naive implementation recomputing all pair counts each iteration is O(n²).
        For efficiency, incrementally update pair counts by only processing words
        that contained the merged pair.
    """
    special_tokens = special_tokens or []
    
    # Read the corpus
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    
    # Build set of "forbidden" substrings from special tokens
    # 这一步见tutorial
    forbidden_substrings = set()
    for special in special_tokens:
        special_bytes = special.encode("utf-8")
        for i in range(2, len(special_bytes) + 1):
            forbidden_substrings.add(special_bytes[:i])
    
    # step1
    vocab = {i:token.encode("utf-8") for i, token in enumerate(special_tokens)} #token is in byte format
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])
    # step2
    tokens = pre_tokenize(text, special_tokens)
    word_freqs = Counter()
    pair_freq = Counter()
    merge = []

    for i in tokens:
        # 1. 必须先转成 bytes，因为 forbidden_substrings 存的是字节
        t_bytes = i.encode("utf-8")
        if any(sub in t_bytes for sub in forbidden_substrings):
            continue

        word_tuple = tuple(bytes([b]) for b in t_bytes)
        word_freqs[word_tuple] += 1

    # word_freqs 长这个样子，比如：
    # word_freqs = {
    #(b'h', b'e', b'l', b'l', b'o'): 2,  # "hello" 出现了 2 次
    #(b'h', b'i'): 1                    # "hi" 出现了 1 次
    #}
    
    # step3 
    for key, value in word_freqs.items():
        for pair in get_pairs(key):
            pair_freq[pair] += value

    # step4
    while len(vocab) < vocab_size:
        if not pair_freq:
            break # no more pairs to merge
        # 4a. Select best pair with deterministic tie-breaking
        best_pair = max(pair_freq, key=lambda p: (pair_freq[p], p))
        # 4b. Create merged token
        merge_pair = best_pair[0] + best_pair[1] # bytes concatenation
        if (best_pair[0] + best_pair[1]) in forbidden_substrings:
            del pair_freq[best_pair]
            continue

        vocab[len(vocab)] = merge_pair
        merge.append(best_pair)
        # 4c. Update word representations
        new_word_freqs = Counter()
        new_pair_freqs = Counter()

        for word, freq in word_freqs.items():
            if best_pair in get_pairs(word):
                new_word = merge_word(word, best_pair)
            else:
                new_word = word
            new_word_freqs[new_word] += freq

            # Recount pairs for this (possibly updated) word
            for pair in get_pairs(new_word):
                new_pair_freqs[pair] += freq

        word_freqs = new_word_freqs
        pair_freq = new_pair_freqs

    return vocab, merge

