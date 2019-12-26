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
	"fmt"
	"io"
	"strings"

	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"

	"github.com/ynqa/wego/pkg/timer"
)

type core struct {
	*corpus.Corpus
	// TODO: more efficient data structure, such as radix tree (trie).
	Document []int
}

func newCore() *core {
	c, _ := corpus.Construct()
	return &core{
		Corpus:   c,
		Document: make([]int, 0),
	}
}

func (c *core) Parse(f io.Reader, toLower bool, minCount int, batchSize int, verbose bool) error {
	fullDoc := make([]int, 0)
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)

	var t *timer.Timer
	if verbose {
		t = timer.NewTimer()
	}
	var i int
	for scanner.Scan() {
		word := scanner.Text()
		if toLower {
			word = strings.ToLower(word)
		}
		// TODO: delete words less than minCount in Corpus.
		c.Add(word)
		wordID, _ := c.Id(word)
		fullDoc = append(fullDoc, wordID)
		if verbose && i%batchSize == 0 {
			fmt.Printf("Read %d words %v\r", i, t.AllElapsed())
		}
		i++
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.Wrap(err, "Unable to complete scanning")
	}
	if verbose {
		fmt.Printf("Read %d words %v\r\n", i, t.AllElapsed())
	}
	for _, d := range fullDoc {
		if c.IDFreq(d) > minCount {
			c.Document = append(c.Document, d)
		}
	}
	if verbose {
		fmt.Printf("Filter words less than minCount=%d > documentSize=%d\n", minCount, len(c.Document))
	}
	return nil
}
