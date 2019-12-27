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

## Lexvec

### Usage

```
Lexvec: Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations

Usage:
  wego lexvec [flags]

Flags:
      --batchSize int       interval word size to update learning rate (default 10000)
  -d, --dimension int       dimension of word vector (default 10)
  -h, --help                help for lexvec
      --initlr float        initial learning rate (default 0.025)
  -i, --inputFile string    input file path for corpus (default "example/input.txt")
      --iter int            number of iteration (default 15)
      --lower               whether the words on corpus convert to lowercase or not
      --min-count int       lower limit to filter rare words (default 5)
  -o, --outputFile string   output file path to save word vectors (default "example/word_vectors.txt")
      --prof                profiling mode to check the performances
      --rel string          relation type for counting co-occurrence. One of ppmi|pmi|co|logco (default "ppmi")
      --sample int          negative sample size(for negative sampling only) (default 5)
      --save-vec string     save vector type. One of: normal|add (default "normal")
      --smooth float        smoothing value (default 0.75)
      --theta float         lower limit of learning rate (lr >= initlr * theta) (default 0.0001)
      --thread int          number of goroutine (default 12)
      --verbose             verbose mode
  -w, --window int          context window size (default 5)
```

# Search

Similarity search between word vectors.

## Usage

```
Search similar words

Usage:
  wego search [flags]

Examples:
  wego search -i example/word_vectors.txt microsoft

Flags:
  -h, --help               help for search
  -i, --inputFile string   input file path for trained word vector (default "example/input.txt")
  -r, --rank int           how many the most similar words will be displayed (default 10)
```

## Example

```
$ go run wego.go search -i example/word_vectors_sg.txt microsoft
  RANK |    WORD    | SIMILARITY
+------+------------+------------+
     1 | apple      |   0.994008
     2 | operating  |   0.992855
     3 | versions   |   0.992800
     4 | ibm        |   0.992232
     5 | os         |   0.989174
     6 | computers  |   0.988998
     7 | machines   |   0.988804
     8 | dvd        |   0.988732
     9 | cd         |   0.988259
    10 | compatible |   0.988200
```

# REPL for search

Similarity search between word vectors with REPL mode.

## Usage

```
Search similar words with REPL mode

Usage:
  wego repl [flags]

Examples:
  wego repl -i example/word_vectors.txt
  >> apple + banana
  ...

Flags:
  -h, --help               help for repl
  -i, --inputFile string   input file path for trained word vector (default "example/word_vectors.txt")
  -r, --rank int           how many the most similar words will be displayed (default 10)
```

## Example

Now, it is able to use `+`, `-` for arithmetic operations.

```
$ go run wego.go repl -i example/word_vectors_sg.txt
>> a + b
  RANK |  WORD   | SIMILARITY
+------+---------+------------+
     1 | phi     |   0.907975
     2 | q       |   0.904593
     3 | mathbf  |   0.903066
     4 | cdot    |   0.902205
     5 | b       |   0.901952
     6 | becomes |   0.900346
     7 | int     |   0.898680
     8 | z       |   0.897895
     9 | named   |   0.896480
    10 | v       |   0.895456
```
