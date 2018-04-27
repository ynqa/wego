# Model

## Word2Vec

Word2Vec is composed of the following modules:

Model:
- Skip-Gram
- CBOW

Optimizer:
- Hierarchical Softmax
- Negative Sampling

### Usage

```
Embed words using word2vec

Usage:
  word-embedding word2vec [flags]

Flags:
      --batchSize int       Set the batch size to update learning rate (default 10000)
  -d, --dimension int       Set the dimension of word vector (default 10)
  -h, --help                Help for word2vec
      --initlr float        Set the initial learning rate (default 0.025)
  -i, --inputFile string    Set the input file path to load corpus (default "example/input.txt")
      --iter int            Set the iteration (default 15)
      --lower               Whether the words on corpus convert to lowercase or not
      --maxDepth int        Set the number of times to track huffman tree, max-depth=0 means to track full path from root to word (using only hierarchical softmax)
      --min-count int       Set the min count to filter rare words (default 5)
      --model string        Set the model of Word2Vec. One of: cbow|skip-gram (default "cbow")
      --optimizer string    Set the optimizer of Word2Vec. One of: hs|ns (default "hs")
  -o, --outputFile string   Set the output file path to save word vectors (default "example/word_vectors.txt")
      --prof                Profiling mode to check the performances
      --sample int          Set the number of the samples as negative (using only negative sampling) (default 5)
      --theta float         Set the lower limit of learning rate (lr >= initlr * theta) (default 0.0001)
      --thread int          Set number of parallel (default 8)
      --threshold float     Set the threshold for subsampling (default 0.001)
      --verbose             Verbose mode
  -w, --window int          Set the context window size (default 5)
```

## GloVe

GloVe is weighted matrix factorization model for co-occurrence map between words.

### Usage

```
Embed words using glove

Usage:
  word-embedding glove [flags]

Flags:
      --alpha float         Set alpha (default 0.75)
  -d, --dimension int       Set the dimension of word vector (default 10)
  -h, --help                Help for glove
      --initlr float        Set the initial learning rate (default 0.025)
  -i, --inputFile string    Set the input file path to load corpus (default "example/input.txt")
      --iter int            Set the iteration (default 15)
      --lower               Whether the words on corpus convert to lowercase or not
      --min-count int       Set the min count to filter rare words (default 5)
  -o, --outputFile string   Set the output file path to save word vectors (default "example/word_vectors.txt")
      --prof                Profiling mode to check the performances
      --solver string       Set the solver of GloVe. One of: sgd|adagrad (default "sgd")
      --thread int          Set number of parallel (default 8)
      --verbose             Verbose mode
  -w, --window int          Set the context window size (default 5)
      --xmax int            Set xmax (default 100)
```