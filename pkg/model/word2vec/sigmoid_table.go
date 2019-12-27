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
	"math"
)

type sigmoidTable struct {
	expTable     []float64
	expTableSize int
	maxExp       float64
	cache        float64
}

func newSigmoidTable() *sigmoidTable {
	s := new(sigmoidTable)
	s.expTableSize = 1000
	s.maxExp = 6.0
	s.cache = float64(s.expTableSize) / s.maxExp / 2.0
	s.expTable = make([]float64, s.expTableSize)
	for i := 0; i < s.expTableSize; i++ {
		expval := math.Exp((float64(i)/float64(s.expTableSize)*2. - 1.) * s.maxExp)
		s.expTable[i] = expval / (expval + 1.)
	}
	return s
}

// sigmoid returns: f(x) = (x + max_exp) * (exp_table_size / max_exp / 2)
// If you set x to over |max_exp|, it raises index out of range error.
func (s *sigmoidTable) sigmoid(x float64) float64 {
	return s.expTable[int((x+s.maxExp)*s.cache)]
}
