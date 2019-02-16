# Word Embedding in Go

[![Build Status](https://travis-ci.org/ynqa/wego.svg?branch=master)](https://travis-ci.org/ynqa/wego)
[![GoDoc](https://godoc.org/github.com/ynqa/wego?status.svg)](https://godoc.org/github.com/ynqa/wego)
[![Go Report Card](https://goreportcard.com/badge/github.com/ynqa/wego)](https://goreportcard.com/report/github.com/ynqa/wego)

Wego is the implementations for word embedding (a.k.a word representation) models in Go. [Word embedding](https://en.wikipedia.org/wiki/Word_embedding) makes word's meaning, structure, and concept mapping into vector space with low dimension.  For representative instance:

```
Vector("King") - Vector("Man") + Vector("Woman") = Vector("Queen")
```

Like this example, models generate word vectors that could calculate word meaning by arithmetic operations for other vectors.

Wego provides CLI that includes not only training model for embedding but also similarity search between words.

## Models

🎃 Word2Vec: Distributed Representations of Words and Phrases and their Compositionality [[pdf]](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

🎃 GloVe: Global Vectors for Word Representation [[pdf]](http://nlp.stanford.edu/pubs/glove.pdf)

🎃 LexVec: Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations [[pdf]](http://anthology.aclweb.org/P16-2068)

## Why Go?

[Data Science in Go](https://speakerdeck.com/chewxy/data-science-in-go) @chewxy

## Installation

```
$ go get -u github.com/ynqa/wego
$ bin/wego -h
```

## Demo

Run the following command, and start to download [text8](http://mattmahoney.net/dc/textdata) corpus and train them by Word2Vec.

```
$ sh demo.sh
```

## Usage

```
Usage:
  wego [flags]
  wego [command]

Available Commands:
  glove       GloVe: Global Vectors for Word Representation
  help        Help about any command
  lexvec      Lexvec: Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations
  repl        Search similar words with REPL mode
  search      Search similar words
  word2vec    Word2Vec: Continuous Bag-of-Words and Skip-gram model

Flags:
  -h, --help   help for wego
```

For more information about each sub-command, see below:
[word2vec](./model/README.md#Word2Vec), [glove](./model/README.md#GloVe), [lexvec](./model/README.md#Lexvec), [search](./search/README.md), [repl](./repl/README.md)

## File I/O

### Input 
Input corpus requires the format that is divided by space between words like [text8](http://mattmahoney.net/dc/textdata) since wego parse with `scanner.Split(bufio.ScanWords)`.

###  Output
Wego outputs a .txt file that is described word vector is subject to the following format:

```
<word> <value1> <value2> ...
```

## Example

It's also able to train word vectors using wego APIs. Examples are as follows.

```go
package main

import (
	"os"

	"github.com/ynqa/wego/builder"
	"github.com/ynqa/wego/model/word2vec"
)

func main() {
	b := builder.NewWord2vecBuilder()

	b.Dimension(10).
		Window(5).
		Model(word2vec.CBOW).
		Optimizer(word2vec.NEGATIVE_SAMPLING).
		NegativeSampleSize(5).
		Verbose()

	m, err := b.Build()
	if err != nil {
		// Failed to build word2vec.
	}

	input, _ := os.Open("text8")

	// Start to Train.
	if err = m.Train(input); err != nil {
		// Failed to train by word2vec.
	}

	// Save word vectors to a text file.
	m.Save("example.txt")
}
```
