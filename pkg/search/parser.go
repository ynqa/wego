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

type StoreFunc func(string, []float64, int)

func ParseAll(f io.Reader, store StoreFunc) error {
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, " ") {
			continue
		}
		word, vec, dim, err := parse(line)
		if err != nil {
			return err
		}
		store(word, vec, dim)
	}
	if err := s.Err(); err != nil && err != io.EOF {
		return errors.Wrapf(err, "Failed to scan %v", f)
	}
	return nil
}

func parse(line string) (string, []float64, int, error) {
	sep := strings.Fields(line)
	word := sep[0]
	elems := sep[1:]
	dim := len(elems)
	vec := make([]float64, dim)
	for k, elem := range elems {
		val, err := strconv.ParseFloat(elem, 64)
		if err != nil {
			return "", nil, 0, err
		}
		vec[k] = val
	}
	return word, vec, dim, nil
}
