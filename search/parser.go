// Copyright © 2019 Makoto Ito
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

package search

import (
	"bufio"
	"io"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

type Parser struct {
	f io.ReadCloser
}

func NewParser(f io.ReadCloser) *Parser {
	return &Parser{
		f: f,
	}
}

type StoreFunc func(string, []float64)

func (p *Parser) ParseAll(store StoreFunc) error {
	defer p.f.Close()

	s := bufio.NewScanner(p.f)
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, " ") {
			continue
		}
		word, vec, err := p.parse(line)
		if err != nil {
			return err
		}
		store(word, vec)
	}
	if err := s.Err(); err != nil && err != io.EOF {
		return errors.Wrapf(err, "Failed to scan %s", p.f)
	}
	return nil
}

func (p *Parser) parse(line string) (string, []float64, error) {
	sep := strings.Fields(line)
	word := sep[0]
	elems := sep[1:]
	vec := make([]float64, len(elems))
	for k, elem := range elems {
		val, err := strconv.ParseFloat(elem, 64)
		if err != nil {
			return "", nil, err
		}
		vec[k] = val
	}
	return word, vec, nil
}
