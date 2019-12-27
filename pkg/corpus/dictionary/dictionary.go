package dictionary

// inspired by
// - https://github.com/chewxy/lingo/blob/master/corpus/corpus.go
// - https://github.com/RaRe-Technologies/gensim/blob/3.8.1/gensim/corpora/dictionary.py

type Dictionary struct {
	word2id map[string]int
	id2word []string

	cfs []int

	maxid int
}

func New() *Dictionary {
	return &Dictionary{
		word2id: make(map[string]int),
		id2word: make([]string, 0),

		cfs: make([]int, 0),
	}
}

func (d *Dictionary) Len() int {
	return d.maxid
}

func (d *Dictionary) ID(word string) (int, bool) {
	id, ok := d.word2id[word]
	return id, ok
}

func (d *Dictionary) WordFreq(word string) int {
	id, ok := d.word2id[word]
	if !ok {
		return 0
	}
	return d.cfs[id]
}

func (d *Dictionary) Word(id int) (string, bool) {
	if id >= d.maxid {
		return "", false
	}
	return d.id2word[id], true
}

func (d *Dictionary) IDFreq(id int) int {
	if id >= d.maxid {
		return 0
	}
	return d.cfs[id]
}

func (d *Dictionary) Add(words ...string) {
	for _, word := range words {
		if id, ok := d.word2id[word]; ok {
			d.cfs[id]++
		} else {
			d.word2id[word] = d.maxid
			d.id2word = append(d.id2word, word)
			d.cfs = append(d.cfs, 1)
			d.maxid++
		}
	}
}
