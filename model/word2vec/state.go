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
	"runtime"
	"strings"
	"sync"

	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/word-embedding/model"
)

// State stores all common configs for Word2Vec models.
type State struct {
	*model.Config
	*corpus.Corpus
	emb *Embedding

	Opt                Optimizer
	SubsampleThreshold float64
	BatchSize          int
	Theta              float64

	ignoredWords        int
	trainedWords        int
	currentLearningRate float64
	progress            *pb.ProgressBar

	// concurrency stuff
	cwsCh   chan struct{}
	lrMutex sync.Mutex
}

// NewState creates *NewState.
func NewState(config *model.Config, opt Optimizer,
	subsampleThreshold, theta float64, batchSize int) *State {
	return &State{
		Config:             config,
		Corpus:             corpus.New(),
		Opt:                opt,
		SubsampleThreshold: subsampleThreshold,
		BatchSize:          batchSize,
		Theta:              theta,

		currentLearningRate: config.InitLearningRate,
		cwsCh:               make(chan struct{}),
	}
}

// Preprocess scans the corpus once before Train to count the word frequency.
func (s *State) Preprocess(f io.ReadSeeker) (io.ReadCloser, error) {
	defer func() {
		s.emb = newEmbedding(s.Corpus.Size(), s.Dimension)
		s.Opt.Init(s.Corpus, s.Dimension)
	}()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		line := scanner.Text()
		if s.Lower {
			line = strings.ToLower(line)
		}

		words := strings.Fields(line)
		for _, word := range words {
			s.Add(word)
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return nil, errors.Wrap(err, "Unable to complete scanning")
	}

	if _, err := f.Seek(0, 0); err != nil {
		return nil, errors.Wrap(err, "Unable to rewind file")
	}
	return f.(io.ReadCloser), nil
}

// Trainer trains a corpus. It assumes that Preprocess() has already been called
func (s *State) Trainer(f io.ReadCloser, trainOne func(wordIDs []int, wordIndex int, lr float64) error) error {
	s.startTraining()
	defer func() {
		s.endTraining()
		f.Close()
	}()

	go s.incrementDoneWord()

	errChan := make(chan error, 1)
	sema := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	var buffered int
	line := make([]string, 0, s.BatchSize)

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		if buffered < s.BatchSize {
			line = append(line, scanner.Text())
			buffered++
			continue
		}

		if s.Lower {
			lower(line)
		}

		wg.Add(1)
		wordIDs := s.toIDs(line)
		go s.trainOneBatch(wordIDs, &wg, sema, errChan, trainOne)
		buffered = 0
		line = line[:0]
	}
	wg.Wait()
	close(errChan)
	if err := <-errChan; err != nil {
		return errors.Wrap(err, "Unable to complete training.")
	}

	// Leftover processing
	if buffered > 0 {
		if s.Lower {
			lower(line)
		}

		wordIDs := s.toIDs(line)
		if err := s.trainRemainderBatch(wordIDs, trainOne); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}

	return nil
}

func (s *State) trainOneBatch(wordIDs []int, wg *sync.WaitGroup, sema chan struct{}, errCh chan error, trainOne func(wordIDs []int, wordIndex int, lr float64) error) {
	defer wg.Done()
	sema <- struct{}{} // get lock
	for i, w := range wordIDs {
		s.progress.Increment()

		r := rand.Float64()
		p := s.subsampleRate(w)

		if p < r {
			s.ignoredWords++
			continue
		}

		s.lrMutex.Lock()
		lr := s.currentLearningRate
		s.lrMutex.Unlock()

		if err := trainOne(wordIDs, i, lr); err != nil {
			select {
			case errCh <- err:
			default:
				// dies silent death
			}
			<-sema // release
			return
		}
		s.cwsCh <- struct{}{} // increment wordsize

	}
	<-sema // release
}

func (s *State) trainRemainderBatch(wordIDs []int, trainOne func(wordIDs []int, wordIndex int, lr float64) error) error {
	for i, w := range wordIDs {
		s.progress.Increment()

		r := rand.Float64()
		p := s.subsampleRate(w)

		if p < r {
			s.ignoredWords++
			continue
		}

		s.lrMutex.Lock()
		lr := s.currentLearningRate
		s.lrMutex.Unlock()

		if err := trainOne(wordIDs, i, lr); err != nil {
			return err
		}
		s.cwsCh <- struct{}{} // increment wordsize
	}
	return nil
}

func (s *State) incrementDoneWord() {
	for range s.cwsCh {
		s.trainedWords++
		if s.trainedWords%s.BatchSize == 0 {
			s.lrMutex.Lock()
			s.currentLearningRate = computeLearningRate(s.InitLearningRate, s.Theta, s.trainedWords, s.TotalFreq())
			s.lrMutex.Unlock()
		}
	}
}

// Save saves the word vector to outputFile.
func (s *State) Save(outputPath string) error {
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

	vs := bytes.NewBuffer(make([]byte, 0))

	for i := 0; i < s.Size(); i++ {
		word, _ := s.Word(i)
		vs.WriteString(fmt.Sprintf("%v ", word))
		vs.WriteString(fmt.Sprintf("%v\n", formatTensor(s.emb.m[i])))
	}

	w.WriteString(fmt.Sprintf("%v", vs.String()))

	return nil
}

func (s *State) startTraining() {
	s.progress = pb.New(s.TotalFreq()).SetWidth(80)
	s.progress.Start()
}

func (s *State) endTraining() {
	s.progress.Finish()
}

func (s *State) subsampleRate(wordID int) float64 {
	z := float64(s.IDFreq(wordID)) / float64(s.TotalFreq())
	return (math.Sqrt(z/s.SubsampleThreshold) + 1.0) * s.SubsampleThreshold / z
}

func (s *State) toIDs(words []string) []int {
	retVal := make([]int, len(words))
	for i, w := range words {
		retVal[i], _ = s.Id(w)
	}
	return retVal
}
