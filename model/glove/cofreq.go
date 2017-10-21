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

// CofreqMap stores the co-frequency between word-word.
type CofreqMap map[Pair]float64

// Pair stores the co-occurrence pair words.
type Pair struct {
	l1 string
	l2 string
}

// PairWithFreq stores the co-occurrence pair words.
type PairWithFreq struct {
	l1 int
	l2 int

	f           float64
	coefficient float64
}

func (c CofreqMap) update(l1, l2 string, distance float64) {
	if l1 == l2 {
		return
	}

	pp := Pair{
		l1: l1,
		l2: l2,
	}

	c[pp] += 1.0 / distance
}

func (c CofreqMap) toList(cps *corpus.Corpus, xmax int, alpha float64) []PairWithFreq {
	lst := make([]PairWithFreq, len(c))
	shuffle := rand.Perm(len(c))

	i := 0
	for p, f := range c {
		l1, _ := cps.Id(p.l1)
		l2, _ := cps.Id(p.l2)

		coefficient := 1.0
		if f < float64(xmax) {
			coefficient = math.Pow(f/float64(xmax), alpha)
		}

		lst[shuffle[i]] = PairWithFreq{
			l1:          l1,
			l2:          l2,
			f:           f,
			coefficient: coefficient,
		}
		i++
	}

	return lst
}
