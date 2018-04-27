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

package corpus

import (
	"bufio"
	"io"
	"strings"

	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"
)

type core struct {
	*corpus.Corpus
	// TODO: more efficient data structure, such as radix tree (trie).
	document []int
}

func newCore() *core {
	c, _ := corpus.Construct()
	return &core{
		Corpus:   c,
		document: make([]int, 0),
	}
}

// Document returns list of word id.
func (c *core) Document() []int {
	return c.document
}

func (c *core) parse(f io.ReadCloser, toLower bool, minCount int) error {
	fullDoc := make([]int, 0)
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		word := scanner.Text()
		if toLower {
			word = strings.ToLower(word)
		}
		c.Add(word)
		wordID, _ := c.Id(word)
		fullDoc = append(fullDoc, wordID)
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}
	for _, d := range fullDoc {
		if c.IDFreq(d) > minCount {
			c.document = append(c.document, d)
		}
	}
	return nil
}
