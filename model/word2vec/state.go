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

	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"
	pb "gopkg.in/cheggaaa/pb.v1"

	"github.com/chewxy/word-embedding/model"
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

	ignoreWord          int
	currentWordSize     int
	currentLearningRate float64
	progress            *pb.ProgressBar
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
	}
}

// Preprocess scans the corpus once before Train to count the word frequency.
func (s *State) Preprocess(f io.ReadSeeker) (io.ReadCloser, error) {
	defer func() {
		s.emb = NewTensor(s.Corpus.Size(), s.Dimension)
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
func (s *State) Trainer(f io.ReadCloser, trainOne func(wordIDs []int, wordIndex int) error) error {
	s.startTraining()
	defer func() {
		s.endTraining()
		f.Close()
	}()

	var current int
	var line string

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		if current < s.BatchSize {
			line += scanner.Text()
			line += " "
			current++
			continue
		}

		if s.Lower {
			line = strings.ToLower(line)
		}

		words := strings.Fields(line)
		wordIDs := s.toIDs(words)
		if err := s.trainOneLine(wordIDs, trainOne); err != nil {
			return err
		}

		current = 0
		line = ""
	}

	// Leftover processing
	if current > 0 {
		if s.Lower {
			line = strings.ToLower(line)
		}

		words := strings.Fields(line)
		wordIDs := s.toIDs(words)
		if err := s.trainOneLine(wordIDs, trainOne); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}

	return nil
}

func (s *State) trainOneLine(wordIDs []int, trainOne func(wordIDs []int, wordIndex int) error) error {
	for i, w := range wordIDs {
		s.progress.Increment()

		r := rand.Float64()
		p := s.subsampleRate(w)

		if p < r {
			s.ignoreWord++
			continue
		}

		trainOne(wordIDs, i)

		s.currentWordSize++
		if s.currentWordSize%s.BatchSize == 0 {
			s.currentLearningRate =
				updateLearningRate(s.InitLearningRate, s.Theta,
					s.currentWordSize, s.TotalFreq())
		}
	}
	return nil
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
		vs.WriteString(fmt.Sprintf("%v\n", s.emb.m[i]))
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
