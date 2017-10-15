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

package model

import (
	"math/rand"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Type stores gorgonia dtype and engine.
type Type struct {
	D tensor.Dtype
	E tensor.Engine
}

// NewType create *Type.
func NewType(t string) (*Type, error) {
	switch t {
	case "float64":
		return &Type{
			D: tensor.Float64,
			E: tensor.Float64Engine{},
		}, nil
	case "float32":
		return &Type{
			D: tensor.Float32,
			E: tensor.Float32Engine{},
		}, nil
	default:
		return nil, errors.Errorf("Dtype is expected one of float64|float32, but actual: %v", t)
	}
}

// RandomTensor create a tensor with shape.
func (t *Type) RandomTensor(shape ...int) tensor.Tensor {
	ref := tensor.New(tensor.Of(t.D), tensor.WithShape(shape...), tensor.WithEngine(t.E))
	switch t.D {
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
