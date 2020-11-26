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

package co

import (
	"fmt"
	"math"

	"github.com/pkg/errors"

	"github.com/ynqa/wego/pkg/corpus/cooccurrence/encode"
)

type CountType = string

const (
	Increment CountType = "inc"
	Proximity CountType = "prox"
)

func invalidCountTypeError(typ CountType) error {
	return fmt.Errorf("invalid relation type: %s not in %s|%s", typ, Increment, Proximity)
}

type Cooccurrence struct {
	typ CountType

	ma map[uint64]float64
}

func New(typ CountType) (*Cooccurrence, error) {
	if typ != Increment && typ != Proximity {
		return nil, invalidCountTypeError(typ)
	}
	return &Cooccurrence{
		typ: typ,

		ma: make(map[uint64]float64),
	}, nil
}

func (c *Cooccurrence) EncodedMatrix() map[uint64]float64 {
	return c.ma
}

func (c *Cooccurrence) Add(left, right int) error {
	enc := encode.EncodeBigram(uint64(left), uint64(right))
	var val float64
	switch c.typ {
	case Increment:
		val = 1
	case Proximity:
		div := left - right
		if div == 0 {
			return errors.Errorf("Divide by zero on counting co-occurrence")
		}
		val = 1. / math.Abs(float64(div))
	default:
		return invalidCountTypeError(c.typ)
	}
	c.ma[enc] += val
	return nil
}
