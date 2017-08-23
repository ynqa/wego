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
	"math"
	"math/rand"
)

// Vector stores the elements in native array.
type Vector []float64

// NewVector creates a Vector.
func NewVector(dimension int) Vector {
	v := make(Vector, dimension)
	return v
}

// NewRandomizedVector create a vector with the random elements.
func NewRandomizedVector(dimension int) Vector {
	v := NewVector(dimension)
	for i := 0; i < dimension; i++ {
		v[i] = (rand.Float64() - 0.5) / float64(dimension)
	}
	return v
}

// Zeros alters the vector to zero elements.
func (v Vector) Zeros() {
	for i := 0; i < len(v); i++ {
		v[i] = 0.
	}
}

// UnsafeAdd adds given vector to the vector.
func (v Vector) UnsafeAdd(vv Vector) {
	for i := 0; i < len(v); i++ {
		v[i] += vv[i]
	}
}

// Mul returns the vector multiplied by given scalar.
func (v Vector) Mul(f float64) Vector {
	res := NewVector(len(v))
	for i := 0; i < len(v); i++ {
		res[i] = f * v[i]
	}
	return res
}

// Inner returns the inner production between vectors.
func (v Vector) Inner(vv Vector) (f float64) {
	for i := 0; i < len(v); i++ {
		f += v[i] * vv[i]
	}
	return
}

// Norm returns the norm of vector.
func (v Vector) Norm() (f float64) {
	for _, val := range v {
		f += val * val
	}
	f = math.Sqrt(f)
	return
}

// Cosine returns the cosine similarity between vectors.
func (v Vector) Cosine(vv Vector) float64 {
	return v.Inner(vv) / (v.Norm() * vv.Norm())
}

func (v Vector) String() string {
	vs := bytes.NewBuffer(make([]byte, 0))
	for key, value := range v {
		vs.WriteString(fmt.Sprintf("%d:%f ", (key + 1), value))
	}
	str := vs.String()
	return str
}
