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
	doc io.Reader

	dic        *dictionary.Dictionary
	cooc       *co.Cooccurrence
	maxLen     int
	indexedDoc []int

	toLower bool
	filters cpsutil.Filters
}

func New(doc io.Reader, toLower bool, maxCount, minCount int) corpus.Corpus {
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

func (c *Corpus) LoadForDictionary() error {
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

	return nil
}

func (c *Corpus) LoadForCooccurrence(typ co.CountType, window int) (err error) {
	c.cooc, err = co.New(typ)
	if err != nil {
		return
	}

	for i := 0; i < len(c.indexedDoc); i++ {
		for j := i + 1; j < len(c.indexedDoc) && j <= i+window; j++ {
			if err = c.cooc.Add(c.indexedDoc[i], c.indexedDoc[j]); err != nil {
				return
			}
		}
	}

	return
}
