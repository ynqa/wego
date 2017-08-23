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

// CBOW is a piece of Word2Vec model.
type CBOW struct {
	*State

	sum  vector.Vector
	pool vector.Vector
}

// NewCBOW creates *CBOW
func NewCBOW(s *State) *CBOW {
	return &CBOW{
		State: s,
		sum:   vector.NewVector(s.Dimension),
		pool:  vector.NewVector(s.Dimension),
	}
}

// Train call Trainer with CBOW trainOne.
func (c *CBOW) Train(f io.ReadCloser) error {
	return c.Trainer(f, c.trainOne)
}

func (c *CBOW) trainOne(wordIDs []int, wordIndex int) error {
	targetID := wordIDs[wordIndex]
	f := func(g func(contextID int)) {
		shr := nextRandom(c.Window)
		for a := shr; a < c.Window*2+1-shr; a++ {
			if a != c.Window {
				c := wordIndex - c.Window + a
				if c < 0 || c >= len(wordIDs) {
					continue
				}
				contextID := wordIDs[c]
				g(contextID)
			}
		}
	}

	c.sum.Zeros()
	initSum := func(contextID int) {
		c.sum.UnsafeAdd(c.Tensor[contextID])
	}
	f(initSum)

	c.pool.Zeros()
	if err := c.Opt.Update(targetID, c.sum, c.pool, c.currentLearningRate); err != nil {
		return err
	}

	updateContext := func(contextID int) {
		c.Tensor[contextID].UnsafeAdd(c.pool)
	}
	f(updateContext)
	return nil
}
