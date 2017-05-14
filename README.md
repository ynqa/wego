# Word Embedding in Golang

[![Build Status](https://travis-ci.org/ynqa/word-embedding.svg?branch=master)](https://travis-ci.org/ynqa/word-embedding)
[![GoDoc](https://godoc.org/github.com/ynqa/word-embedding?status.svg)](https://godoc.org/github.com/ynqa/word-embedding)
[![Go Report Card](https://goreportcard.com/badge/github.com/ynqa/word-embedding)](https://goreportcard.com/report/github.com/ynqa/word-embedding)

This is an implementation of word embedding (also referred to as word representation) models in Golang.

## Details

Word embedding makes words' meaning, structure, and concept mapping into vector space (and low dimension). For representative instance:

```
Vector("king") - Vector("Man") + Vector("Woman") = Vector("Queen")
```

Like this example, it could calculate word meaning by arithmetic operations between vectors.

## Features
Listed models for word embedding, and checked it already implemented.

### Models
- [x] Word2vec
  - Distributed Representations of Words and Phrases
and their Compositionality [[pdf]](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [ ] GloVe
  - GloVe: Global Vectors for Word Representation [[pdf]](http://nlp.stanford.edu/pubs/glove.pdf)
- [ ] SPPMI-MF
  - Neural Word Embedding as Implicit Matrix Factorization [[pdf]](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)

## Installation

```
$ go get -u github.com/ynqa/word-embedding
$ bin/word-embedding -h
```

## Usage

```
The tools embedding words into vector space

Usage:
  word-embedding [flags]
  word-embedding [command]

Available Commands:
  sim         Estimate the similarity between words
  word2vec    Embed words using word2vec
```

## File I/O
- Input
  - Given a text is composed of one-sentence per one-line, ideally.
- Output
  - Output a file is like [libsvm](https://github.com/cjlin1/libsvm) format:
  ```
  <word> <index1>:<value1> <index2>:<value2> ...
  ```

## References
- Just see it for more deep comprehension:
  - Improving Distributional Similarity
with Lessons Learned from Word Embeddings [[pdf]](http://www.aclweb.org/anthology/Q15-1016)
  - Don’t count, predict! A systematic comparison of
context-counting vs. context-predicting semantic vectors [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.648.8023&rep=rep1&type=pdf)
