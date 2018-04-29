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

// SkipGram behaviors as one of Word2vec solver.
type SkipGram struct {
	pools chan []float64

	dimension int
	window    int
}

// NewSkipGram creates *SkipGram
func NewSkipGram(dimension, window, threadSize int) *SkipGram {
	pools := make(chan []float64, threadSize)
	for i := 0; i < threadSize; i++ {
		pools <- make([]float64, dimension)
	}
	return &SkipGram{
		pools: pools,

		dimension: dimension,
		window:    window,
	}
}

func (s *SkipGram) trainOne(document []int, wordIndex int, wordVector []float64, lr float64, optimizer Optimizer) {
	pool := <-s.pools
	word := document[wordIndex]
	shrinkage := model.NextRandom(s.window)
	for a := shrinkage; a < s.window*2+1-shrinkage; a++ {
		if a == s.window {
			continue
		}
		c := wordIndex - s.window + a
		if c < 0 || c >= len(document) {
			continue
		}
		context := document[c]
		for i := 0; i < s.dimension; i++ {
			pool[i] = 0.0
		}
		optimizer.update(word, lr, wordVector[context*s.dimension:context*s.dimension+s.dimension], pool)
		for i := 0; i < s.dimension; i++ {
			wordVector[context*s.dimension+i] += pool[i]
		}
	}
	s.pools <- pool
}
