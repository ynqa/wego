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

package vector

import (
	"bytes"
	"fmt"
	"math/rand"
)

type Vector []float64

func NewVector(dim int) Vector {
	v := make(Vector, dim)
	return v
}

func NewRandomizedVector(dim int) Vector {
	v := NewVector(dim)
	for i := 0; i < dim; i++ {
		v[i] = (rand.Float64() - 0.5) / float64(dim)
	}
	return v
}

func (v1 Vector) Inner(v2 Vector) float64 {
	f := 0.
	for i := 0; i < len(v1); i++ {
		f += v1[i] * v2[i]
	}
	return f
}

func (v Vector) String() string {
	vs := bytes.NewBuffer(make([]byte, 0))
	for key, value := range v {
		vs.WriteString(fmt.Sprintf("%d:%f ", (key + 1), value))
	}
	str := vs.String()
	return str
}
