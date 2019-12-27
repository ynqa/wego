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

package item

import (
	"bufio"
	"io"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

type Item struct {
	Word   string
	Dim    int
	Vector []float64
}

func (i Item) Validate() error {
	if i.Word == "" {
		return errors.New("Word is empty")
	} else if i.Dim == 0 || len(i.Vector) == 0 {
		return errors.Errorf("Dim of %s is zero", i.Word)
	} else if i.Dim != len(i.Vector) {
		return errors.Errorf("Dim and length of Vector must be same, Dim=%d, len(Vec)=%d", i.Dim, len(i.Vector))
	}
	return nil
}

type ItemOp func(Item) error

func Parse(r io.Reader, op ItemOp) error {
	s := bufio.NewScanner(r)
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, " ") {
			continue
		}
		item, err := ParseLine(line)
		if err != nil {
			return err
		}
		if err := op(item); err != nil {
			return err
		}
	}
	if err := s.Err(); err != nil && err != io.EOF {
		return errors.Wrapf(err, "failed to scan")
	}
	return nil
}

func ParseLine(line string) (Item, error) {
	slice := strings.Fields(line)
	if len(slice) < 2 {
		return Item{}, errors.New("Must be over 2 lenghth for word and vector elems")
	}
	word := slice[0]
	elems := slice[1:]
	dim := len(elems)

	vec := make([]float64, dim)
	for k, elem := range elems {
		val, err := strconv.ParseFloat(elem, 64)
		if err != nil {
			return Item{}, err
		}
		vec[k] = val
	}
	return Item{
		Word:   word,
		Dim:    dim,
		Vector: vec,
	}, nil
}
