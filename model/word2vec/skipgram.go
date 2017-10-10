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
	"runtime"

	"github.com/chewxy/gorgonia/tensor"
)

// SkipGram is a piece of Word2Vec model.
type SkipGram struct {
	*State

	pools chan tensor.Tensor
}

// NewSkipGram creates *SkipGram
func NewSkipGram(s *State) *SkipGram {
	maxprocs := runtime.NumCPU()
	pool := make(chan tensor.Tensor, maxprocs)
	for i := 0; i < maxprocs; i++ {
		pool <- tensor.New(tensor.WithShape(s.Dimension), tensor.Of(s.Dtype.T), tensor.WithEngine(s.Dtype.E))
	}
	return &SkipGram{
		State: s,
		pools: pool,
	}
}

// Train call Trainer with SkipGram trainOne.
func (s *SkipGram) Train(f io.ReadCloser) error {
	return s.Trainer(f, s.trainOne)
}

func (s *SkipGram) trainOne(wordIDs []int, wordIndex int, lr float64) error {
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

		pool.Zero()
		if err := s.opt.Update(s.Dtype, targetID, s.emb.m[contextID].Tensor, pool, lr); err != nil {
			return err
		}
		tensor.Add(s.emb.m[contextID].Tensor, pool, tensor.UseUnsafe())
	}
	s.pools <- pool
	return nil
}
