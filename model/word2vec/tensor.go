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
	"github.com/ynqa/word-embedding/vector"
)

// Tensor has the vector for each word.
type Tensor []vector.Vector

// NewTensor creates Tensor with the shape of (vocabulary, dimension).
func NewTensor(vocabulary, dimension int) Tensor {
	tensor := make([]vector.Vector, vocabulary)
	for i := 0; i < vocabulary; i++ {
		tensor[i] = vector.NewRandomizedVector(dimension)
	}
	return tensor
}
