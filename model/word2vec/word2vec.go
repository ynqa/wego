// Copyright © 2017 Makoto Ito
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package word2vec

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/word-embedding/corpus"
	"github.com/ynqa/word-embedding/model"
)

// Word2Vec stores all common configs for Word2Vec models.
type Word2Vec struct {
	*model.Config
	*corpus.Word2VecCorpus

	mod Model
	opt Optimizer

	subsampleThreshold float64
	subSamples         []float64
	batchSize          int
	theta              float64

	vector []float64

	indexPerThread      []int
	trainedWords        int
	currentLearningRate float64
	// concurrency stuff
	cwsCh chan struct{}

	progress *pb.ProgressBar
}

// NewWord2Vec creates *Word2Vec.
func NewWord2Vec(f io.ReadCloser, config *model.Config, mod Model, opt Optimizer,
	subsampleThreshold, theta float64, batchSize int) *Word2Vec {
	c := corpus.NewWord2VecCorpus(f, config.ToLower, config.MinCount)
	word2vec := &Word2Vec{
		Config:         config,
		Word2VecCorpus: c,

		mod: mod,
		opt: opt,

		subsampleThreshold:  subsampleThreshold,
		batchSize:           batchSize,
		theta:               theta,
		currentLearningRate: config.InitLearningRate,

		indexPerThread: make([]int, config.Thread+1),
		cwsCh:          make(chan struct{}),
	}
	word2vec.initialize()
	return word2vec
}

func (w *Word2Vec) initialize() {
	// Store subsumple before training.
	w.subSamples = make([]float64, w.Corpus.Size())
	for i := 0; i < w.Corpus.Size(); i++ {
		z := float64(w.IDFreq(i)) / float64(w.TotalFreq())
		w.subSamples[i] = (math.Sqrt(z/w.subsampleThreshold) + 1.0) *
			w.subsampleThreshold / z
	}

	// Initialize word vector.
	w.vector = make([]float64, w.Corpus.Size()*w.Dimension)
	for i := 0; i < w.Corpus.Size()*w.Dimension; i++ {
		w.vector[i] = (rand.Float64() - 0.5) / float64(w.Dimension)
	}

	// Initialize optimizer.
	w.opt.initialize(w.Word2VecCorpus, w.Dimension)
}

// Train trains a corpus.
func (w *Word2Vec) Train() error {
	doc := w.Document()
	if len(doc) <= 0 {
		return errors.New("No words for training")
	}

	numWords := len(doc)
	w.indexPerThread[0] = 0
	w.indexPerThread[w.Thread] = numWords
	for i := 1; i < w.Thread; i++ {
		w.indexPerThread[i] = w.indexPerThread[i-1] + int(math.Trunc(float64((numWords+i)/w.Thread)))
	}

	for i := 1; i <= w.Iteration; i++ {
		if w.Verbose {
			fmt.Printf("%d-th:\n", i)
			w.progress = pb.New(len(doc)).SetWidth(80)
			w.progress.Start()
		}
		go w.incrementDoneWord()

		sema := make(chan struct{}, w.Thread)
		var wg sync.WaitGroup

		for j := 0; j < w.Thread; j++ {
			wg.Add(1)
			go w.trainPerThread(doc[w.indexPerThread[j]:w.indexPerThread[j+1]], &wg, sema, w.mod.trainOne)
		}
		wg.Wait()
		if w.Verbose {
			w.progress.Finish()
		}
	}
	return nil
}

func (w *Word2Vec) trainPerThread(doc []int, wg *sync.WaitGroup, sema chan struct{},
	trainOne func(wordIDs []int, wordIndex int, wordVector []float64, lr float64, optimizer Optimizer)) {
	defer wg.Done()
	sema <- struct{}{} // get lock
	for i, d := range doc {
		if w.Verbose {
			w.progress.Increment()
		}

		r := rand.Float64()
		p := w.subSamples[d]
		if p < r {
			continue
		}
		lr := w.currentLearningRate
		trainOne(doc, i, w.vector, lr, w.opt)
		w.cwsCh <- struct{}{} // increment wordsize
	}
	<-sema // release
}

func (w *Word2Vec) incrementDoneWord() {
	for range w.cwsCh {
		w.trainedWords++
		if w.trainedWords%w.batchSize == 0 {
			w.currentLearningRate = w.updateLearningRate()
		}
	}
}

// Save saves the word vector to outputFile.
func (w *Word2Vec) Save(outputPath string) error {
	extractDir := func(path string) string {
		e := strings.Split(path, "/")
		return strings.Join(e[:len(e)-1], "/")
	}

	dir := extractDir(outputPath)

	if err := os.MkdirAll("."+string(filepath.Separator)+dir, 0777); err != nil {
		return err
	}

	file, err := os.Create(outputPath)

	if err != nil {
		return err
	}
	wr := bufio.NewWriter(file)

	defer func() {
		wr.Flush()
		file.Close()
	}()

	var buf bytes.Buffer
	for i := 0; i < w.Size(); i++ {
		word, _ := w.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < w.Dimension; j++ {
			fmt.Fprintf(&buf, "%f ", w.vector[i*w.Dimension+j])
		}
		fmt.Fprintln(&buf)
	}

	wr.WriteString(fmt.Sprintf("%v", buf.String()))

	return nil
}

func (w *Word2Vec) updateLearningRate() float64 {
	lr := w.InitLearningRate * (1.0 - float64(w.trainedWords)/float64(w.TotalFreq()))
	if lr < w.InitLearningRate*w.theta {
		lr = w.InitLearningRate * w.theta
	}
	return lr
}
