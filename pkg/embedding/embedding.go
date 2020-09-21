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

package embedding

import (
	"bufio"
	"io"
	"strconv"
	"strings"

	"github.com/pkg/errors"

	"github.com/ynqa/wego/pkg/embedding/embutil"
)

type Embedding struct {
	Word   string
	Dim    int
	Vector []float64
	Norm   float64
}

func (e Embedding) Validate() error {
	if e.Word == "" {
		return errors.New("Word is empty")
	} else if e.Dim == 0 || len(e.Vector) == 0 {
		return errors.Errorf("Dim of %s is zero", e.Word)
	} else if e.Dim != len(e.Vector) {
		return errors.Errorf("Dim and length of Vector must be same, Dim=%d, len(Vec)=%d", e.Dim, len(e.Vector))
	}
	return nil
}

type Embeddings []Embedding

func (embs Embeddings) Empty() bool {
	return len(embs) == 0
}

func (embs Embeddings) Find(word string) (Embedding, bool) {
	for _, emb := range embs {
		if word == emb.Word {
			return emb, true
		}
	}
	return Embedding{}, false
}

func (embs Embeddings) Validate() error {
	if len(embs) > 0 {
		dim := embs[0].Dim
		for _, emb := range embs {
			if dim != emb.Dim {
				return errors.Errorf("dimension for all vectors must be the same: %d but got %d", dim, emb.Dim)
			}
		}
	}
	return nil
}

func Load(r io.Reader) (Embeddings, error) {
	var embs Embeddings
	if err := parse(r, func(emb Embedding) error {
		if err := emb.Validate(); err != nil {
			return err
		}
		embs = append(embs, emb)
		return nil
	}); err != nil {
		return nil, err
	}
	return embs, nil
}

func parse(r io.Reader, op func(Embedding) error) error {
	s := bufio.NewScanner(r)
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, " ") {
			continue
		}
		emb, err := parseLine(line)
		if err != nil {
			return err
		}
		if err := op(emb); err != nil {
			return err
		}
	}
	if err := s.Err(); err != nil && err != io.EOF {
		return errors.Wrapf(err, "failed to scan")
	}
	return nil
}

func parseLine(line string) (Embedding, error) {
	slice := strings.Fields(line)
	if len(slice) < 2 {
		return Embedding{}, errors.New("Must be over 2 lenghth for word and vector elems")
	}
	word := slice[0]
	vector := slice[1:]
	dim := len(vector)

	vec := make([]float64, dim)
	for k, elem := range vector {
		val, err := strconv.ParseFloat(elem, 64)
		if err != nil {
			return Embedding{}, err
		}
		vec[k] = val
	}
	return Embedding{
		Word:   word,
		Dim:    dim,
		Vector: vec,
		Norm:   embutil.Norm(vec),
	}, nil
}
