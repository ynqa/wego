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
Word2Vec: Continuous Bag-of-Words and Skip-gram model

Usage:
  wego word2vec [flags]

Flags:
      --batchSize int       interval word size to update learning rate (default 10000)
  -d, --dimension int       dimension of word vector (default 10)
  -h, --help                help for word2vec
      --initlr float        initial learning rate (default 0.025)
  -i, --inputFile string    input file path for corpus (default "example/input.txt")
      --iter int            number of iteration (default 15)
      --lower               whether the words on corpus convert to lowercase or not
      --maxDepth int        times to track huffman tree, max-depth=0 means to track full path from root to word (for hierarchical softmax only)
      --min-count int       lower limit to filter rare words (default 5)
      --model string        which model does it use? one of: cbow|skip-gram (default "cbow")
      --optimizer string    which optimizer does it use? one of: hs|ns (default "hs")
  -o, --outputFile string   output file path to save word vectors (default "example/word_vectors.txt")
      --prof                profiling mode to check the performances
      --sample int          negative sample size(for negative sampling only) (default 5)
      --theta float         lower limit of learning rate (lr >= initlr * theta) (default 0.0001)
      --thread int          number of goroutine (default 8)
      --threshold float     threshold for subsampling (default 0.001)
      --verbose             verbose mode
  -w, --window int          context window size (default 5)
```

## GloVe

GloVe is weighted matrix factorization model for co-occurrence map between words.

### Usage

```
GloVe: Global Vectors for Word Representation

Usage:
  wego glove [flags]

Flags:
      --alpha float         exponent of weighting function (default 0.75)
  -d, --dimension int       dimension of word vector (default 10)
  -h, --help                help for glove
      --initlr float        initial learning rate (default 0.025)
  -i, --inputFile string    input file path for corpus (default "example/input.txt")
      --iter int            number of iteration (default 15)
      --lower               whether the words on corpus convert to lowercase or not
      --min-count int       lower limit to filter rare words (default 5)
  -o, --outputFile string   output file path to save word vectors (default "example/word_vectors.txt")
      --prof                profiling mode to check the performances
      --solver string       solver for GloVe objective. One of: sgd|adagrad (default "sgd")
      --thread int          number of goroutine (default 8)
      --verbose             verbose mode
  -w, --window int          context window size (default 5)
      --xmax int            specifying cutoff in weighting function (default 100)
```
