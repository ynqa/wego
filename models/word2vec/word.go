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
	"bytes"
	"fmt"
	"math/rand"

	"github.com/ynqa/word-embedding/utils/fileio"
	"github.com/ynqa/word-embedding/utils/set"
	"github.com/ynqa/word-embedding/utils/vector"
)

type Word struct {
	Vector           vector.Vector
	VectorAsNegative vector.Vector
}
type WordMap map[string]*Word

var keys []string

func NewWordMapFrom(s set.String, vectorDim int, neg bool) WordMap {
	wordMap := make(WordMap)
	keys = make([]string, len(s))

	f := func(b bool) vector.Vector {
		if b {
			return vector.NewVector(vectorDim)
		} else {
			return nil
		}
	}

	i := 0
	for v := range s {
		wordMap[v] = &Word{
			Vector:           vector.NewRandomizedVector(vectorDim),
			VectorAsNegative: f(neg),
		}
		keys[i] = v
		i++
	}
	return wordMap
}

func (w WordMap) GetRandom() (key string, value *Word) {
	l := len(w)
	index := rand.Intn(l)
	key = keys[index]
	value = w[key]
	return
}

func (w WordMap) Save(outputPath string) error {
	return fileio.Save(outputPath, w)
}

func (w WordMap) String() string {
	vs := bytes.NewBuffer(make([]byte, 0))
	for k, v := range w {
		vs.WriteString(fmt.Sprintf("%s ", k))
		vs.WriteString(fmt.Sprintf("%v\n", v.Vector))
	}
	return vs.String()
}
