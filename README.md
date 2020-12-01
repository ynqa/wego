# Word Embeddings in Go

[![Build Status](https://travis-ci.com/ynqa/wego.svg?branch=master)](https://travis-ci.com/ynqa/wego)
[![GoDoc](https://godoc.org/github.com/ynqa/wego?status.svg)](https://godoc.org/github.com/ynqa/wego)
[![Go Report Card](https://goreportcard.com/badge/github.com/ynqa/wego)](https://goreportcard.com/report/github.com/ynqa/wego)

*wego* is the implementations **from scratch** for word embeddings (a.k.a word representation) models in Go.

## What's word embeddings?

[Word embeddings](https://en.wikipedia.org/wiki/Word_embeddings) make words' meaning, structure, and concept mapping into vector space with a low dimension. For representative instance:
```
Vector("King") - Vector("Man") + Vector("Woman") = Vector("Queen")
```
Like this example, the models generate word vectors that could calculate word meaning by arithmetic operations for other vectors.

## Features

The following models to capture the word vectors are supported in *wego*:

- Word2Vec: Distributed Representations of Words and Phrases and their Compositionality [[pdf]](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

- GloVe: Global Vectors for Word Representation [[pdf]](http://nlp.stanford.edu/pubs/glove.pdf)

- LexVec: Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations [[pdf]](http://anthology.aclweb.org/P16-2068)

Also, wego provides nearest neighbor search tools that calculate the distances between word vectors and find the nearest words for the target word. "near" for word vectors means "similar" for words.

Please see the [Usage](#Usage) section if you want to know how to use these for more details.

## Why Go?

Inspired by [Data Science in Go](https://speakerdeck.com/chewxy/data-science-in-go) @chewxy

## Installation

```
$ go get -u github.com/ynqa/wego
$ bin/wego -h
```

## Usage

*wego* provides CLI and Go SDK for word embeddings.

### CLI

```
Usage:
  wego [flags]
  wego [command]

Available Commands:
  console     Console to investigate word vectors
  glove       GloVe: Global Vectors for Word Representation
  help        Help about any command
  lexvec      Lexvec: Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations
  query       Query similar words
  word2vec    Word2Vec: Continuous Bag-of-Words and Skip-gram model
```

### Go SDK

```go
package main

import (
	"os"

	"github.com/ynqa/wego/pkg/model/modelutil/vector"
	"github.com/ynqa/wego/pkg/model/word2vec"
)

func main() {
	model, err := word2vec.New(
		word2vec.Window(5),
		word2vec.Model(word2vec.Cbow),
		word2vec.Optimizer(word2vec.NegativeSampling),
		word2vec.NegativeSampleSize(5),
		word2vec.Verbose(),
	)
	if err != nil {
		// failed to create word2vec.
	}

	input, _ := os.Open("text8")
	if err = model.Train(input); err != nil {
		// failed to train.
	}

	// write word vector.
	model.Save(os.Stdin, vector.Agg)
}
```

## Formats

As training word vectors *wego* requires file format for inputs/outputs.

### Input

Input corpus must be subject to the formats to be divided by space between words like [text8](http://mattmahoney.net/dc/textdata.html).

```
word1 word2 word3 ...
```

###  Output

After training *wego* save the word vectors into a txt file with the following format (`N` is the dimension for word vectors you given):

```
<word> <value_1> <value_2> ... <value_N>
```
