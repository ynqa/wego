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

package fs

import (
	"io"
	"strings"

	"github.com/ynqa/wego/pkg/corpus"
	co "github.com/ynqa/wego/pkg/corpus/cooccurrence"
	"github.com/ynqa/wego/pkg/corpus/cpsutil"
	"github.com/ynqa/wego/pkg/corpus/dictionary"
)

type Corpus struct {
	doc io.ReadSeeker

	dic        *dictionary.Dictionary
	cooc       *co.Cooccurrence
	maxLen     int
	indexedDoc []int

	toLower bool
	filters cpsutil.Filters
}

func New(r io.ReadSeeker, toLower bool, maxCount, minCount int) corpus.Corpus {
	return &Corpus{
		doc:        r,
		dic:        dictionary.New(),
		indexedDoc: make([]int, 0),

		toLower: toLower,
		filters: cpsutil.Filters{
			cpsutil.MaxCount(maxCount),
			cpsutil.MinCount(minCount),
		},
	}
}

func (c *Corpus) IndexedDoc() []int {
	return nil
}

func (c *Corpus) BatchWords(ch chan []int, batchSize int) error {
	cursor, ids := 0, make([]int, batchSize)
	if err := cpsutil.ReadWord(c.doc, func(word string) error {
		if c.toLower {
			word = strings.ToLower(word)
		}

		id, _ := c.dic.ID(word)
		if c.filters.Any(id, c.dic) {
			return nil
		}

		ids[cursor] = id
		cursor++
		if cursor == batchSize {
			ch <- ids
			cursor, ids = 0, make([]int, batchSize)
		}
		return nil
	}); err != nil {
		return err
	}

	// send left words
	ch <- ids[:cursor]
	close(ch)
	return nil
}

func (c *Corpus) Dictionary() *dictionary.Dictionary {
	return c.dic
}

func (c *Corpus) Cooccurrence() *co.Cooccurrence {
	return c.cooc
}

func (c *Corpus) Len() int {
	return c.maxLen
}

func (c *Corpus) Load(fn func(int), with *corpus.WithCooccurrence) error {
	if err := cpsutil.ReadWord(c.doc, func(word string) error {
		if c.toLower {
			word = strings.ToLower(word)
		}

		c.dic.Add(word)
		c.maxLen++
		fn(c.maxLen)

		return nil
	}); err != nil {
		return err
	}

	var err error
	if with != nil {
		c.cooc, err = co.New(with.CountType)
		if err != nil {
			return err
		}

		if err = cpsutil.ReadWordWithForwardContext(
			c.doc, with.Window, func(w1, w2 string) error {
				id1, _ := c.dic.ID(w1)
				id2, _ := c.dic.ID(w2)
				if err := c.cooc.Add(id1, id2); err != nil {
					return err
				}
				return nil
			}); err != nil {
			return err
		}
	}

	return nil
}
