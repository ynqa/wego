// Copyright © 2020 wego authors
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

package repl

import (
	"github.com/pkg/errors"
)

type Operator func(float64, float64) float64

func elementWise(v1, v2 []float64, op Operator) ([]float64, error) {
	if len(v1) != len(v2) {
		return nil, errors.Errorf("Both lengths of vector must be the same, got %d and %d", len(v1), len(v2))
	}
	v := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		v[i] = op(v1[i], v2[i])
	}
	return v, nil
}

func add(v1, v2 []float64) ([]float64, error) {
	return elementWise(v1, v2, addOp())
}

func addOp() Operator {
	return Operator(func(x, y float64) float64 {
		return x + y
	})
}

func sub(v1, v2 []float64) ([]float64, error) {
	return elementWise(v1, v2, subOp())
}

func subOp() Operator {
	return Operator(func(x, y float64) float64 {
		return x - y
	})
}
