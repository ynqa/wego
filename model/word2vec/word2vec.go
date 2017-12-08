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

	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/word-embedding/model"
)

// Word2Vec stores all common configs for Word2Vec models.
type Word2Vec struct {
	*model.Config
	*corpus.Corpus

	mod Model
	opt Optimizer

	subsampleThreshold float64
	batchSize          int
	theta              float64

	vector []float64

	trainedWords        int
	currentLearningRate float64
	progress            *pb.ProgressBar

	// concurrency stuff
	cwsCh chan struct{}
}

// NewWord2Vec creates *Word2Vec.
func NewWord2Vec(config *model.Config, mod Model, opt Optimizer,
	subsampleThreshold, theta float64, batchSize int) *Word2Vec {
	return &Word2Vec{
		Config: config,
		Corpus: corpus.New(),

		mod: mod,
		opt: opt,

		subsampleThreshold: subsampleThreshold,
		batchSize:          batchSize,
		theta:              theta,

		currentLearningRate: config.InitLearningRate,
		cwsCh:               make(chan struct{}),
	}
}

// Preprocess scans the corpus once before Train to count the word frequency.
func (w *Word2Vec) Preprocess(f io.ReadSeeker) (io.ReadCloser, error) {
	defer func() {
		w.vector = make([]float64, w.Corpus.Size()*w.Dimension)
		for i := 0; i < w.Corpus.Size()*w.Dimension; i++ {
			w.vector[i] = (rand.Float64() - 0.5) / float64(w.Dimension)
		}
		w.opt.init(w.Corpus, w.Dimension)
	}()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		word := scanner.Text()
		if w.ToLower {
			word = strings.ToLower(word)
		}
		w.Add(word)
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return nil, errors.Wrap(err, "Unable to complete scanning")
	}

	if _, err := f.Seek(0, 0); err != nil {
		return nil, errors.Wrap(err, "Unable to rewind file")
	}
	return f.(io.ReadCloser), nil
}

// Train trains a corpus. It assumes that Preprocess() has already been called
func (w *Word2Vec) Train(f io.ReadCloser) error {
	if w.Verbose {
		w.startTraining()
		defer w.endTraining()
	}

	defer f.Close()

	go w.incrementDoneWord()

	sema := make(chan struct{}, w.Thread)
	var wg sync.WaitGroup

	buffered := 0
	wordIDs := make([]int, w.batchSize)

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		word := scanner.Text()
		if w.ToLower {
			word = strings.ToLower(word)
		}
		wordID, _ := w.Id(word)
		wordIDs[buffered] = wordID

		if buffered < w.batchSize-1 {
			buffered++
			continue
		}

		wg.Add(1)
		go w.trainOneBatch(wordIDs, &wg, sema, w.mod.trainOne)
		wordIDs = make([]int, w.batchSize)
		buffered = 0
	}

	// Leftover processing
	if buffered > 0 {
		wg.Add(1)
		go w.trainOneBatch(wordIDs[:buffered], &wg, sema, w.mod.trainOne)
	}

	wg.Wait()

	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}

	return nil
}

func (w *Word2Vec) trainOneBatch(wordIDs []int, wg *sync.WaitGroup, sema chan struct{},
	trainOne func(wordIDs []int, wordIndex int, wordVector []float64, lr float64, optimizer Optimizer)) {

	defer wg.Done()
	sema <- struct{}{} // get lock
	for i, wordID := range wordIDs {
		if w.Verbose {
			w.progress.Increment()
		}

		r := rand.Float64()
		p := w.subsampleRate(wordID)

		if p < r {
			continue
		}

		lr := w.currentLearningRate
		trainOne(wordIDs, i, w.vector, lr, w.opt)
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

func (w *Word2Vec) startTraining() {
	w.progress = pb.New(w.TotalFreq()).SetWidth(80)
	w.progress.Start()
}

func (w *Word2Vec) endTraining() {
	w.progress.Finish()
}

func (w *Word2Vec) updateLearningRate() float64 {
	lr := w.InitLearningRate * (1.0 - float64(w.trainedWords)/float64(w.TotalFreq()))
	if lr < w.InitLearningRate*w.theta {
		lr = w.InitLearningRate * w.theta
	}
	return lr
}

func (w *Word2Vec) subsampleRate(wordID int) float64 {
	z := float64(w.IDFreq(wordID)) / float64(w.TotalFreq())
	return (math.Sqrt(z/w.subsampleThreshold) + 1.0) * w.subsampleThreshold / z
}
