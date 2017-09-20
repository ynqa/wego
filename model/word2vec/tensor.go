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
	"github.com/chewxy/gorgonia/tensor"
	"github.com/ynqa/word-embedding/model"
)

var dtype tensor.Dtype = tensor.Float64
var eng tensor.Engine = tensor.Float64Engine{}

// s implements a single valued slice
type s int

func (a s) Start() int { return int(a) }
func (a s) End() int   { return int(a + 1) }
func (a s) Step() int  { return 1 }

// Embedding represents a word embedding. It holds a Tensor, and preslices it for additional performance gains.
type Embedding struct {
	ref tensor.Tensor // hold a reference
	m   []*model.SyncTensor
}

func newEmbedding(vocabulary, dimension int) *Embedding {
	ref := randomTensor(dtype, eng, vocabulary, dimension)

	// preslice all the things!
	m := make([]*model.SyncTensor, vocabulary)
	for i := 0; i < vocabulary; i++ {
		slice, _ := ref.Slice(s(i))
		m[i] = &model.SyncTensor{Tensor: slice}
	}

	return &Embedding{
		ref: ref,
		m:   m,
	}
}
