# Word2Vec

Word2Vec is the generic term below modules:

```
model:
- Skip-Gram
- CBOW

optimizer:
- Hierarchical Softmax
- Negative Sampling
```

In training, select one `model` and one `optimizer` above. `model` and `optimizer` represent architecture of objective and the way of approximating its function respectively.

## Features

- [x] Skip-Gram
- [x] CBOW
- [x] Hierarchical Softmax
- [x] Negative Sampling
- [ ] Subsampling
- [x] Update learning rate in training

## Usage

```
Embed words using word2vec

Usage:
  word-embedding word2vec [flags]

Flags:
  -d, --dimension int      Set word vector dimension size (default 10)
  -i, --input string       Input file path for learning (default "example/input.txt")
      --lr float           Set init learning rate (default 0.025)
      --max-depth int      Set number of times to track huffman tree, max-depth=0 means tracking full path (using only hierarchical softmax)
      --model string       Set model from: skip-gram|cbow (default "cbow")
      --negative int       Set number of negative samplings (using only negative sampling) (default 5)
      --optimizer string   Set optimizer from: hs|ns (default "hs")
  -o, --output string      Output file path for each learned word vector (default "example/word_vectors.txt")
  -w, --window int         Set window size (default 5)
```
