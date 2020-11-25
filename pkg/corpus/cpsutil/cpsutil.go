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

package cpsutil

import (
	"bufio"
	"io"

	"github.com/ynqa/wego/pkg/corpus/dictionary"
)

func scanner(r io.Reader) *bufio.Scanner {
	s := bufio.NewScanner(r)
	s.Split(bufio.ScanWords)
	return s
}

func ReadWord(r io.Reader, fn func(string) error) error {
	scanner := scanner(r)
	for scanner.Scan() {
		if err := fn(scanner.Text()); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return err
	}

	return nil
}

func ReadWordWithForwardContext(r io.Reader, n int, fn func(string, string) error) error {
	scanner := scanner(r)
	var (
		axis string
		ws   []string = make([]string, n)
	)
	postFn := func() error {
		for _, w := range ws {
			if err := fn(axis, w); err != nil {
				return err
			}
		}
		return nil
	}
	for {
		if axis == "" {
			if !scanner.Scan() {
				break
			}
			axis = scanner.Text()
			for i := 0; i < n; i++ {
				if !scanner.Scan() {
					break
				}
				ws[i] = scanner.Text()
			}
		} else {
			axis = ws[0]
			ws = ws[1:]
			if !scanner.Scan() {
				break
			}
			ws = append(ws, scanner.Text())
		}
		if err := postFn(); err != nil {
			return err
		}
	}
	if err := postFn(); err != nil {
		return err
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return err
	}

	return nil
}

type Filters []FilterFn

func (f Filters) Any(id int, dic *dictionary.Dictionary) bool {
	var b bool
	for _, fn := range f {
		b = b || fn(id, dic)
	}
	return b
}

type FilterFn func(int, *dictionary.Dictionary) bool

func MaxCount(v int) FilterFn {
	return FilterFn(func(id int, dic *dictionary.Dictionary) bool {
		return 0 < v && v < dic.IDFreq(id)
	})
}

func MinCount(v int) FilterFn {
	return FilterFn(func(id int, dic *dictionary.Dictionary) bool {
		return 0 <= v && dic.IDFreq(id) < v
	})
}
