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

package fileio

import (
	"bufio"
	"io"
	"os"
)

const (
	defaultBatch   = 10000
	defaultBufSize = 4096
)

var (
	rr *bufio.Reader
)

// Load reads the text in input file.
func Load(inputFile string, f func(line []string)) error {
	return loadBatch(inputFile, defaultBatch, f)
}

func loadBatch(inputFile string, batch int, f func(sentences []string)) error {
	file, err := os.Open(inputFile)

	if err != nil {
		return err
	}
	defer file.Close()

	rr = bufio.NewReader(file)

	j := 0
	for {
		lines := make([]string, 0)
		for i := 1; i <= batch; i++ {
			line, err := readLine()
			if err == io.EOF {
				f(lines)
				return nil
			} else if err != nil {
				return err
			}
			lines = append(lines, line)
		}
		f(lines)
		j++
	}
}

func readLine() (string, error) {
	buf := make([]byte, 0, defaultBufSize)
	for {
		line, isPrefix, err := rr.ReadLine()
		if err != nil {
			return "", err
		}

		buf = append(buf, line...)
		if !isPrefix {
			break
		}
	}
	return string(buf), nil
}
