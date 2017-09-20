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
	"strings"

	"github.com/chewxy/gorgonia/tensor"
)

var next uint64 = 1

// Linear congruential generator like rand.Intn(window)
func nextRandom(window int) int {
	next = next*uint64(25214903917) + 11
	return int(next % uint64(window))
}

func computeLearningRate(initlr, theta float64, currentWords, totalWords int) float64 {
	lr := initlr * (1.0 - float64(currentWords)/float64(totalWords))
	if lr < initlr*theta {
		lr = initlr * theta
	}
	return lr
}

func lower(a []string) {
	for i := range a {
		a[i] = strings.ToLower(a[i])
	}
}

func formatTensor(t tensor.Tensor) string {
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

func randomTensor(dt tensor.Dtype, eng tensor.Engine, shape ...int) tensor.Tensor {
	ref := tensor.New(tensor.Of(dt), tensor.WithShape(shape...), tensor.WithEngine(eng))
	switch dtype {
	case tensor.Float64:
		dat := ref.Data().([]float64)
		for i := range dat {
			dat[i] = (rand.Float64() - 0.5) / float64(shape[len(shape)-1])
		}
	case tensor.Float32:
		dat := ref.Data().([]float32)
		for i := range dat {
			dat[i] = (rand.Float32() - 0.5) / float32(shape[len(shape)-1])
		}
	}
	return ref
}
