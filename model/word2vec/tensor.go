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

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/word-embedding/model"
)

var dtype tensor.Dtype = tensor.Float64
var eng tensor.Engine = tensor.Float64Engine{}

type s int

func (a s) Start() int { return int(a) }
func (a s) End() int   { return int(a + 1) }
func (a s) Step() int  { return 1 }

type Embedding struct {
	ref tensor.Tensor // hold a reference
	m   []*model.SyncTensor
}

func newEmbedding(vocabulary, dimension int) *Embedding {
	ref := tensor.New(tensor.Of(dtype), tensor.WithShape(vocabulary, dimension), tensor.WithEngine(eng))
	switch dtype {
	case tensor.Float64:
		dat := ref.Data().([]float64)
		for i := range dat {
			dat[i] = (rand.Float64() - 0.5) / float64(dimension)
		}
	case tensor.Float32:
		dat := ref.Data().([]float32)
		for i := range dat {
			dat[i] = (rand.Float32() - 0.5) / float32(dimension)
		}
	}

	// preslice all the things!
	// m := make([]tensor.Tensor, vocabulary)
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

func format(t tensor.Tensor) string {
	var buf bytes.Buffer
	switch data := t.Data().(type) {
	case []float64:
		for i, v := range data {
			fmt.Fprintf(&buf, "%d:%f ", i+1, v)
		}
	case []float32:
		for i, v := range data {
			fmt.Fprintf(&buf, "%d:%f ", i+1, v)
		}
	}
	return buf.String()
}
