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
)

// SkipGram is a piece of Word2Vec model.
type SkipGram struct {
	*State

	pools chan []float64
}

// NewSkipGram creates *SkipGram
func NewSkipGram(s *State) *SkipGram {
	pools := make(chan []float64, s.Thread)
	for i := 0; i < s.Thread; i++ {
		pools <- make([]float64, s.Dimension)
	}
	return &SkipGram{
		State: s,
		pools: pools,
	}
}

// Train call Trainer with SkipGram trainOne.
func (s *SkipGram) Train(f io.ReadCloser) error {
	return s.Trainer(f, s.trainOne)
}

func (s *SkipGram) trainOne(wordIDs []int, wordIndex int, lr float64) {
	// grab poolvector from pool
	pool := <-s.pools
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

		for i := 0; i < s.Dimension; i++ {
			pool[i] = 0.0
		}

		s.opt.Update(targetID, s.vector[contextID*s.Dimension:contextID*s.Dimension+s.Dimension], pool, lr)

		for i := 0; i < s.Dimension; i++ {
			s.vector[contextID*s.Dimension+i] += pool[i]
		}
	}
	s.pools <- pool
}
