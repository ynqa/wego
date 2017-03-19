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

package model

import (
	"github.com/ynqa/word-embedding/models"
	"github.com/ynqa/word-embedding/models/word2vec"
	"github.com/ynqa/word-embedding/utils/vector"
)

type SkipGram struct {
	models.Common
}

func (s SkipGram) Train(words []string, index int, opt func(target string, contentOrSumVector, poolVector vector.Vector)) {
	target := words[index]
	for a := 0; a < s.Common.Window*2+1; a++ {
		if a == s.Common.Window {
			continue
		}
		c := index - s.Common.Window + a
		if c < 0 || c >= len(words) {
			continue
		}
		context := words[c]

		pool := vector.NewVector(s.Common.Dimension)
		opt(target, word2vec.GlobalWordMap[context].Vector, pool)

		for d := 0; d < s.Common.Dimension; d++ {
			word2vec.GlobalWordMap[context].Vector[d] += pool[d]
		}
	}
	printTrace(word2vec.GetWords())
}
