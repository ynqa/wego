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
	"io"

	"github.com/ynqa/word-embedding/vector"
)

// SkipGram is a piece of Word2Vec model.
type SkipGram struct {
	*State

	pool vector.Vector
}

// NewSkipGram creates *SkipGram
func NewSkipGram(s *State) *SkipGram {
	return &SkipGram{
		State: s,
		pool:  vector.NewVector(s.Dimension),
	}
}

// Train call Trainer with SkipGram trainOne.
func (s *SkipGram) Train(f io.ReadCloser) error {
	return s.Trainer(f, s.trainOne)
}

func (s *SkipGram) trainOne(wordIDs []int, wordIndex int) error {
	targetID := wordIDs[wordIndex]
	shr := nextRandom(s.Window)
	for a := shr; a < s.Window*2+1-shr; a++ {
		if a == s.Window {
			continue
		}
		c := wordIndex - s.Window + a
		if c < 0 || c >= len(wordIDs) {
			continue
		}
		contextID := wordIDs[c]

		s.pool.Zeros()
		if err := s.Opt.Update(targetID, s.Tensor[contextID],
			s.pool, s.currentLearningRate); err != nil {
			return err
		}

		s.Tensor[contextID].UnsafeAdd(s.pool)
	}
	return nil
}
