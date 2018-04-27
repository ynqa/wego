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

package corpus

import (
	"io"

	"github.com/ynqa/word-embedding/corpus/co"
)

// CountModelCorpus stores corpus and co-occurrences for words.
type CountModelCorpus struct {
	*core
	cooccurrence map[uint64]float64
}

// NewCountModelCorpus creates *CountModelCorpus.
func NewCountModelCorpus(f io.ReadCloser, toLower bool, minCount, window int,
	incf func(l1idx, l2idx int) float64) *CountModelCorpus {
	countModelCorpus := &CountModelCorpus{
		core:         newCore(),
		cooccurrence: make(map[uint64]float64),
	}
	countModelCorpus.parse(f, toLower, minCount)
	countModelCorpus.build(window, incf)
	return countModelCorpus
}

// Cooccurrence returns co-occurrence map for words.
func (cc *CountModelCorpus) Cooccurrence() map[uint64]float64 {
	return cc.cooccurrence
}

func (cc *CountModelCorpus) build(window int, incf func(l1idx, l2idx int) float64) {
	for i := 0; i < len(cc.document); i++ {
		for j := i + 1; j <= i+window; j++ {
			if j >= len(cc.document) {
				continue
			}
			f := incf(i, j)
			cc.cooccurrence[co.EncodeBigram(uint64(cc.document[i]), uint64(cc.document[j]))] += f
			cc.cooccurrence[co.EncodeBigram(uint64(cc.document[j]), uint64(cc.document[i]))] += f
		}
	}
}
