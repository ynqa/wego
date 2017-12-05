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

package glove

import (
	"math"
	"math/rand"

	"github.com/chewxy/lingo/corpus"
)

func encodeBigram(l1, l2 uint64) uint64 {
	return l1 | (l2 << 32)
}

func decodeBigram(pid uint64) (uint64, uint64) {
	f := pid >> 32
	return pid - (f << 32), f
}

// CofreqMap stores the co-frequency between word-word.
type CofreqMap map[uint64]float64

// PairWithFreq stores the co-frequency pair words.
type PairWithFreq struct {
	l1, l2 int

	f           float64
	coefficient float64
}

func (c CofreqMap) update(l1, l2 int, distance float64) {
	c[encodeBigram(uint64(l1), uint64(l2))] += 1.0 / distance
}

func (c CofreqMap) toList(cps *corpus.Corpus, xmax int, alpha float64, minCount int) []PairWithFreq {
	for p := range c {
		ul1, ul2 := decodeBigram(p)
		if cps.IDFreq(int(ul1)) < minCount || cps.IDFreq(int(ul2)) < minCount {
			delete(c, p)
		}
	}

	lst := make([]PairWithFreq, len(c))
	shuffle := rand.Perm(len(c))

	i := 0
	for p, f := range c {
		coefficient := 1.0
		if f < float64(xmax) {
			coefficient = math.Pow(f/float64(xmax), alpha)
		}

		ul1, ul2 := decodeBigram(p)
		lst[shuffle[i]] = PairWithFreq{
			l1:          int(ul1),
			l2:          int(ul2),
			f:           math.Log(f),
			coefficient: coefficient,
		}
		i++
	}

	return lst
}
