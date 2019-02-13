// Copyright © 2019 Makoto Ito
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

package lexvec

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

	"github.com/ynqa/wego/corpus"
	"github.com/ynqa/wego/corpus/co"
	"github.com/ynqa/wego/model"
)

type LexvecOption struct {
	BatchSize          int
	NegativeSampleSize int
	SubSampleThreshold float64
	Theta              float64
	Smooth             float64
	RelationType       corpus.RelationType
}

// Lexvec stores the configs for Lexvec models.
type Lexvec struct {
	*model.Option
	*LexvecOption
	*corpus.CountModelCorpus

	subSamples []float64

	// word pairs.
	pairs corpus.PairMap

	// words' vector.
	vector []float64

	// manage learning rate.
	currentlr        float64
	trained          chan struct{}
	trainedWordCount int

	// data size per thread.
	indexPerThread []int

	// progress bar.
	progress *pb.ProgressBar
}

// NewLexvec create *Lexvec.
func NewLexvec(f io.ReadCloser, option *model.Option, lexvecOption *LexvecOption) (*Lexvec, error) {
	c := corpus.NewCountModelCorpus()
	if err := c.Parse(f, option.ToLower, option.MinCount); err != nil {
		return nil, errors.Wrap(err, "Unable to generate *Lexvec")
	}
	lexvec := &Lexvec{
		Option:           option,
		LexvecOption:     lexvecOption,
		CountModelCorpus: c,

		currentlr: option.Initlr,
		trained:   make(chan struct{}),
	}
	lexvec.initialize()
	return lexvec, nil
}

func (l *Lexvec) initialize() (err error) {
	// Build pairs based on co-occurrence.
	l.pairs, err = l.CountModelCorpus.PairsIntoLexvec(l.Window, l.RelationType, l.Smooth, l.Verbose)

	// Store subsample before training.
	l.subSamples = make([]float64, l.Corpus.Size())
	for i := 0; i < l.Corpus.Size(); i++ {
		z := 1. - math.Sqrt(l.SubSampleThreshold/float64(l.IDFreq(i)))
		if z < 0 {
			z = 0
		}
		l.subSamples[i] = z
	}

	// Initialize word vector.
	vectorSize := l.Corpus.Size() * l.Dimension * 2
	l.vector = make([]float64, vectorSize)
	for i := 0; i < vectorSize; i++ {
		l.vector[i] = (rand.Float64() - 0.5) / float64(l.Dimension)
	}
	return nil
}

// Train trains words' vector on corpus.
func (l *Lexvec) Train() error {
	document := l.Document()
	documentSize := len(document)
	if documentSize <= 0 {
		return errors.New("No words for training")
	}

	l.indexPerThread = model.IndexPerThread(l.ThreadSize, documentSize)

	for i := 1; i <= l.Iteration; i++ {
		if l.Verbose {
			fmt.Printf("Train %d-th:\n", i)
			l.progress = pb.New(documentSize).SetWidth(80)
			l.progress.Start()
		}
		go l.observeLearningRate(i)

		semaphore := make(chan struct{}, l.ThreadSize)
		waitGroup := &sync.WaitGroup{}

		for j := 0; j < l.ThreadSize; j++ {
			waitGroup.Add(1)
			go l.trainPerThread(document[l.indexPerThread[j]:l.indexPerThread[j+1]], semaphore, waitGroup)
		}

		waitGroup.Wait()
		if l.Verbose {
			l.progress.Finish()
		}
	}
	return nil
}

func (l *Lexvec) trainPerThread(document []int, semaphore chan struct{}, waitGroup *sync.WaitGroup) {
	defer func() {
		waitGroup.Done()
		<-semaphore
	}()

	for idx, wordID := range document {
		if l.Verbose {
			l.progress.Increment()
		}

		bernoulliTrial := rand.Float64()
		p := l.subSamples[wordID]
		if p < bernoulliTrial {
			continue
		}
		l.scan(document, idx, l.vector, l.currentlr)
		l.trained <- struct{}{}
	}
}

func (l *Lexvec) scan(document []int, wordIndex int, wordVector []float64, lr float64) {
	word := document[wordIndex]
	l1 := word * l.Dimension
	shrinkage := model.NextRandom(l.Window)
	for a := shrinkage; a < l.Window*2+1-shrinkage; a++ {
		if a == l.Window {
			continue
		}
		c := wordIndex - l.Window + a
		if c < 0 || c >= len(document) {
			continue
		}
		context := document[c]
		l2 := context * l.Dimension
		encoded := co.EncodeBigram(uint64(word), uint64(context))
		l.trainOne(l1, l2, l.pairs[encoded])
		for n := 0; n < l.NegativeSampleSize; n++ {
			sample := model.NextRandom(l.CountModelCorpus.Size())
			encoded := co.EncodeBigram(uint64(word), uint64(sample))
			l2 := (sample + l.CountModelCorpus.Size()) * l.Dimension
			l.trainOne(l1, l2, l.pairs[encoded])
		}
	}
}

func (l *Lexvec) trainOne(l1, l2 int, f float64) {
	var diff float64
	for i := 0; i < l.Dimension; i++ {
		diff += l.vector[l1+i] * l.vector[l2+i]
	}
	diff = (diff - f) * l.currentlr
	for i := 0; i < l.Dimension; i++ {
		t1 := diff * l.vector[l2+i]
		t2 := diff * l.vector[l1+i]
		l.vector[l1+i] -= t1
		l.vector[l2+i] -= t2
	}
}

func (l *Lexvec) observeLearningRate(iteration int) {
	for range l.trained {
		l.trainedWordCount++
		if l.trainedWordCount%l.BatchSize == 0 {
			l.currentlr = l.Initlr *
				(1. - float64(l.trainedWordCount)/
					(float64(l.Corpus.TotalFreq())-float64(iteration)))
			if l.currentlr < l.Initlr*l.Theta {
				l.currentlr = l.Initlr * l.Theta
			}
		}
	}
}

func (l *Lexvec) Save(outputPath string) error {
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
	w := bufio.NewWriter(file)

	defer func() {
		w.Flush()
		file.Close()
	}()

	wordSize := l.CountModelCorpus.Size()
	if l.Verbose {
		fmt.Println("Save:")
		l.progress = pb.New(wordSize).SetWidth(80)
		defer l.progress.Finish()
		l.progress.Start()
	}

	var buf bytes.Buffer
	for i := 0; i < wordSize; i++ {
		word, _ := l.CountModelCorpus.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < l.Dimension; j++ {
			l1 := i*l.Dimension + j
			var v float64
			switch l.SaveVectorType {
			case model.NORMAL:
				v = l.vector[l1]
			case model.ADD:
				l2 := (i+wordSize)*l.Dimension + j
				v = l.vector[l1] + l.vector[l2]
			default:
				return errors.Errorf("Invalid save vector type=%s", l.SaveVectorType)
			}
			fmt.Fprintf(&buf, "%v ", v)
		}
		fmt.Fprintln(&buf)
		if l.Verbose {
			l.progress.Increment()
		}
	}
	w.WriteString(fmt.Sprintf("%v", buf.String()))
	return nil
}
