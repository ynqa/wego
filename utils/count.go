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

package utils

import (
	"github.com/ynqa/word-embedding/utils/set"
)

type FreqMap map[string]int

func NewFreqMap() FreqMap {
	return make(FreqMap)
}

func (f FreqMap) Update(words []string) {
	for _, w := range words {
		f[w] += 1
	}
}

func (f FreqMap) Keys() set.String {
	keys := set.New()
	for k := range f {
		keys.Add(k)
	}
	return keys
}

func (f FreqMap) Terms() int {
	return len(f)
}

func (f FreqMap) Words() (s int) {
	for _, v := range f {
		s += v
	}
	return
}
