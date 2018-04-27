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

// GloveCorpus stores corpus and co-occurrences for words.
type GloveCorpus struct {
	*core
	cooccurrence map[uint64]float64
}

// NewGloveCorpus creates *GloveCorpus.
func NewGloveCorpus(f io.ReadCloser, toLower bool, minCount, window int,
	incf func(l1idx, l2idx int) float64) *GloveCorpus {
	gloveCorpus := &GloveCorpus{
		core:         newCore(),
		cooccurrence: make(map[uint64]float64),
	}
	gloveCorpus.parse(f, toLower, minCount)
	gloveCorpus.build(window, incf)
	return gloveCorpus
}

// Cooccurrence returns co-occurrence map for words.
func (gc *GloveCorpus) Cooccurrence() map[uint64]float64 {
	return gc.cooccurrence
}

func (gc *GloveCorpus) build(window int, incf func(l1idx, l2idx int) float64) {
	for i := 0; i < len(gc.document); i++ {
		for j := i + 1; j <= i+window; j++ {
			if j >= len(gc.document) {
				continue
			}
			f := incf(i, j)
			gc.cooccurrence[co.EncodeBigram(uint64(gc.document[i]), uint64(gc.document[j]))] += f
			gc.cooccurrence[co.EncodeBigram(uint64(gc.document[j]), uint64(gc.document[i]))] += f
		}
	}
}
