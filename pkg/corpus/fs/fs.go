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
	doc io.Reader

	dic        *dictionary.Dictionary
	cooc       *co.Cooccurrence
	maxLen     int
	indexedDoc []int

	toLower bool
	filters cpsutil.Filters
}

func New(r io.Reader, toLower bool, maxCount, minCount int) corpus.Corpus {
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
	err := cpsutil.ReadWord(c.doc, func(word string) error {
		if c.toLower {
			word = strings.ToLower(word)
		}

		id, _ := c.dic.ID(word)
		if c.filters.Any(id, c.dic) {
			return nil
		}

		ids[cursor] = id
		cursor++
		if len(ids) == batchSize {
			ch <- ids
			cursor, ids = 0, make([]int, batchSize)
		}
		return nil
	})
	if err != nil {
		return err
	}

	// send left words
	ch <- ids[:cursor]
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
		c.maxLen++

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

	if err = cpsutil.ReadWordWithForwardContext(
		c.doc, window, func(w1, w2 string) error {
			id1, _ := c.dic.ID(w1)
			id2, _ := c.dic.ID(w2)
			if err := c.cooc.Add(id1, id2); err != nil {
				return err
			}
			return nil
		}); err != nil {
		return
	}
	return
}
