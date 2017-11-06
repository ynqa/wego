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
	"github.com/ynqa/word-embedding/model"
)

// SkipGram is a piece of Word2Vec model.
type SkipGram struct {
	pools chan []float64

	window    int
	dimension int
}

// NewSkipGram creates *SkipGram
func NewSkipGram(c *model.Config) *SkipGram {
	pools := make(chan []float64, c.Thread)
	for i := 0; i < c.Thread; i++ {
		pools <- make([]float64, c.Dimension)
	}
	return &SkipGram{
		pools: pools,

		window:    c.Window,
		dimension: c.Dimension,
	}
}

func (s *SkipGram) trainOne(wordIDs []int, wordIndex int, wordVector []float64, lr float64, optimizer Optimizer) {
	// grab poolvector from pool
	pool := <-s.pools
	targetID := wordIDs[wordIndex]
	shr := nextRandom(s.window)
	for a := shr; a < s.window*2+1-shr; a++ {
		if a == s.window {
			continue
		}
		c := wordIndex - s.window + a
		if c < 0 || c >= len(wordIDs) {
			continue
		}
		contextID := wordIDs[c]

		for i := 0; i < s.dimension; i++ {
			pool[i] = 0.0
		}

		optimizer.update(targetID, wordVector[contextID*s.dimension:contextID*s.dimension+s.dimension], pool, lr)

		for i := 0; i < s.dimension; i++ {
			wordVector[contextID*s.dimension+i] += pool[i]
		}
	}
	s.pools <- pool
}
