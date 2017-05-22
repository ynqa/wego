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
	"fmt"
	"math"
	"math/rand"

	"github.com/ynqa/word-embedding/models"
	"github.com/ynqa/word-embedding/utils"
	"github.com/ynqa/word-embedding/utils/fileio"
	"gopkg.in/cheggaaa/pb.v1"
)

var ignoredWords int

const (
	batchForUpdateLearningRate = 10000
)

// Word2Vec stores the model, and optimizer.
type Word2Vec struct {
	models.Common
	// Skip-Gram, or CBOW.
	Model Model
	// Hierarchical Softmax, or Negative Sampling.
	Opt                Optimizer
	SubSampleThreshold float64
}

// PreTrain prepares word statistical info for training.
func (w Word2Vec) PreTrain() error {
	return w.Opt.PreTrain()
}

// Run executes training words' vector.
func (w Word2Vec) Run() error {
	w.Opt.InitLearningRate(GetWords())
	progressor := pb.New(GetWords()).SetWidth(80)
	progressor.Start()
	if err := fileio.Load(w.Common.InputFile, utils.ToLowerWords(w.iterator(progressor))); err != nil {
		return err
	}
	progressor.Finish()
	fmt.Printf("Ignored words by subsampling: %d / %d\n", ignoredWords, GetWords())
	return nil
}

func (w Word2Vec) subSampleRate(word string) float64 {
	z := float64(GlobalFreqMap[word]) / float64(GetWords())
	p := (math.Sqrt(z/w.SubSampleThreshold) + 1.0) * w.SubSampleThreshold / z
	return p
}

func (w Word2Vec) iterator(progressor *pb.ProgressBar) func(words []string) {
	return func(words []string) {
		currentWords := 0
		for index, word := range words {
			progressor.Increment()
			r := rand.Float64()
			p := w.subSampleRate(word)
			if p < r {
				ignoredWords++
				continue
			}
			w.Model.Train(words, index, w.Opt.Update)
			currentWords++
			if currentWords%batchForUpdateLearningRate == 0 {
				w.Opt.UpdateLearningRate(currentWords)
			}
		}
	}
}

// Save calls word map itself.
func (w Word2Vec) Save() error {
	return GlobalWordMap.Save(w.Common.OutputFile)
}
