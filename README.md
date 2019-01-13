# Word Embedding in Golang

[![Build Status](https://travis-ci.org/ynqa/wego.svg?branch=master)](https://travis-ci.org/ynqa/wego)
[![GoDoc](https://godoc.org/github.com/ynqa/wego?status.svg)](https://godoc.org/github.com/ynqa/wego)
[![Go Report Card](https://goreportcard.com/badge/github.com/ynqa/wego)](https://goreportcard.com/report/github.com/ynqa/wego)

This is the implementation of word embedding (a.k.a word representation) models in Golang.

## Word Embedding

Word embedding makes word's meaning, structure, and concept mapping into vector space with low dimension. For representative instance:

```
Vector("King") - Vector("Man") + Vector("Woman") = Vector("Queen")
```

Like this example, the models generate the vectors that could calculate word meaning by arithmetic operations for other vectors.

## Models
- [x] Word2Vec
  - Distributed Representations of Words and Phrases
and their Compositionality [[pdf]](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [x] GloVe
  - GloVe: Global Vectors for Word Representation [[pdf]](http://nlp.stanford.edu/pubs/glove.pdf)

and more...

## Installation

```
$ go get -u github.com/ynqa/wego
$ bin/wego -h
```

## Usage

```
tools for embedding words into vector space

Usage:
  wego [flags]
  wego [command]

Available Commands:
  glove       GloVe: Global Vectors for Word Representation
  help        Help about any command
  search      Search similar words
  word2vec    Word2Vec: Continuous Bag-of-Words and Skip-gram model

Flags:
  -h, --help   help for wego

Use "wego [command] --help" for more information about a command.
```

For more information about each sub-command, see below:
- [search](./search/README.md)
- [word2vec](./model/README.md)
- [glove](./model/README.md)

## Demo

Downloading [text8](http://mattmahoney.net/dc/textdata) corpus, and training by Skip-Gram with negative sampling.

```
$ sh demo.sh
```

## Output
Output a file is subject to the following format:

```
<word> <value1> <value2> ...
```
