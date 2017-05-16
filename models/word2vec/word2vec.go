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
	"github.com/ynqa/word-embedding/models"
	"github.com/ynqa/word-embedding/utils/fileio"
	"gopkg.in/cheggaaa/pb.v1"
)

const batchForUpdateLearningRate = 10000

// Word2Vec stores the model, and optimizer.
type Word2Vec struct {
	models.Common
	// Skip-Gram, or CBOW.
	Model Model
	// Hierarchical Softmax, or Negative Sampling.
	Opt Optimizer
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
	if err := fileio.Load(w.Common.InputFile, w.iterator(progressor)); err != nil {
		return err
	}
	progressor.Finish()
	return nil
}

func (w Word2Vec) iterator(progressor *pb.ProgressBar) func(words []string) {
	return func(words []string) {
		currentWords := 0
		for index := range words {
			w.Model.Train(words, index, w.Opt.Update)
			progressor.Increment()
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
