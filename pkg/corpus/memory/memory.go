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

package memory

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

func New(doc io.ReadSeeker, toLower bool, maxCount, minCount int) corpus.Corpus {
	return &Corpus{
		doc:        doc,
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
	var res []int
	for _, id := range c.indexedDoc {
		if c.filters.Any(id, c.dic) {
			continue
		}
		res = append(res, id)
	}
	return res
}

func (c *Corpus) BatchWords(chan []int, int) error {
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

func (c *Corpus) Load(with *corpus.WithCooccurrence) error {
	if err := cpsutil.ReadWord(c.doc, func(word string) error {
		if c.toLower {
			word = strings.ToLower(word)
		}

		c.dic.Add(word)
		id, _ := c.dic.ID(word)
		c.maxLen++
		c.indexedDoc = append(c.indexedDoc, id)

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

		for i := 0; i < len(c.indexedDoc); i++ {
			for j := i + 1; j < len(c.indexedDoc) && j <= i+with.Window; j++ {
				if err = c.cooc.Add(c.indexedDoc[i], c.indexedDoc[j]); err != nil {
					return err
				}
			}
		}
	}

	return nil
}
