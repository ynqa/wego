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

package lexvec

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"math/rand"
	"sync"

	"golang.org/x/sync/semaphore"

	"github.com/ynqa/wego/pkg/clock"
	"github.com/ynqa/wego/pkg/corpus"
	co "github.com/ynqa/wego/pkg/corpus/cooccurrence"
	"github.com/ynqa/wego/pkg/corpus/cooccurrence/encode"
	"github.com/ynqa/wego/pkg/corpus/fs"
	"github.com/ynqa/wego/pkg/corpus/memory"
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/model/modelutil"
	"github.com/ynqa/wego/pkg/model/modelutil/matrix"
	"github.com/ynqa/wego/pkg/model/modelutil/save"
	"github.com/ynqa/wego/pkg/model/modelutil/subsample"
	"github.com/ynqa/wego/pkg/verbose"
)

type lexvec struct {
	opts Options

	corpus corpus.Corpus

	param      *matrix.Matrix
	subsampler *subsample.Subsampler
	currentlr  float64

	verbose *verbose.Verbose
}

func New(opts ...ModelOption) (model.Model, error) {
	options := DefaultOptions()
	for _, fn := range opts {
		fn(&options)
	}

	return NewForOptions(options)
}

func NewForOptions(opts Options) (model.Model, error) {
	// TODO: validate Options
	v := verbose.New(opts.Verbose)
	return &lexvec{
		opts: opts,

		currentlr: opts.Initlr,

		verbose: v,
	}, nil
}

func (l *lexvec) preTrain(r io.Reader) error {
	if l.opts.DocInMemory {
		l.corpus = memory.New(r, l.opts.ToLower, l.opts.MaxCount, l.opts.MinCount)
	} else {
		l.corpus = fs.New(r, l.opts.ToLower, l.opts.MaxCount, l.opts.MinCount)
	}
	if err := l.corpus.LoadForDictionary(); err != nil {
		return err
	}
	if err := l.corpus.LoadForCooccurrence(co.Increment, l.opts.Window); err != nil {
		return err
	}

	dic, dim := l.corpus.Dictionary(), l.opts.Dim

	l.param = matrix.New(
		dic.Len()*2,
		dim,
		func(vec []float64) {
			for i := 0; i < dim; i++ {
				vec[i] = (rand.Float64() - 0.5) / float64(dim)
			}
		},
	)

	l.subsampler = subsample.New(dic, l.opts.SubsampleThreshold)
	return nil
}

func (l *lexvec) Train(r io.Reader) error {
	if err := l.preTrain(r); err != nil {
		return err
	}

	items, err := l.makeItems(l.corpus.Cooccurrence())
	if err != nil {
		return err
	}
	doc := l.corpus.IndexedDoc()
	indexPerThread := modelutil.IndexPerThread(
		l.opts.Goroutines,
		len(doc),
	)

	for i := 1; i <= l.opts.Iter; i++ {
		trained, clk := make(chan struct{}), clock.New()
		go l.observe(trained, clk)

		sem := semaphore.NewWeighted(int64(l.opts.Goroutines))
		wg := &sync.WaitGroup{}

		for i := 0; i < l.opts.Goroutines; i++ {
			wg.Add(1)
			s, e := indexPerThread[i], indexPerThread[i+1]
			go l.trainPerThread(doc[s:e], items, trained, sem, wg)
		}

		wg.Wait()
		close(trained)
	}
	return nil
}

func (l *lexvec) trainPerThread(
	doc []int,
	items map[uint64]float64,
	trained chan struct{},
	sem *semaphore.Weighted,
	wg *sync.WaitGroup,
) error {
	defer func() {
		wg.Done()
		sem.Release(1)
	}()

	if err := sem.Acquire(context.Background(), 1); err != nil {
		return err
	}

	for pos, id := range doc {
		if l.subsampler.Trial(id) {
			l.trainOne(doc, pos, items)
		}
		trained <- struct{}{}
	}

	return nil
}

func (l *lexvec) trainOne(doc []int, pos int, items map[uint64]float64) {
	dic := l.corpus.Dictionary()
	del := modelutil.NextRandom(l.opts.Window)
	for a := del; a < l.opts.Window*2+1-del; a++ {
		if a == l.opts.Window {
			continue
		}
		c := pos - l.opts.Window + a
		if c < 0 || c >= len(doc) {
			continue
		}
		enc := encode.EncodeBigram(uint64(doc[pos]), uint64(doc[c]))
		l.update(doc[pos], doc[c], items[enc])
		for n := 0; n < l.opts.NegativeSampleSize; n++ {
			sample := modelutil.NextRandom(dic.Len())
			enc := encode.EncodeBigram(uint64(doc[pos]), uint64(sample))
			l.update(doc[pos], sample+dic.Len(), items[enc])
		}
	}
}

func (l *lexvec) update(l1, l2 int, f float64) {
	var diff float64
	for i := 0; i < l.opts.Dim; i++ {
		diff += l.param.Slice(l1)[i] * l.param.Slice(l2)[i]
	}
	diff = (diff - f) * l.currentlr
	for i := 0; i < l.opts.Dim; i++ {
		t1 := diff * l.param.Slice(l2)[i]
		t2 := diff * l.param.Slice(l1)[i]
		l.param.Slice(l1)[i] -= t1
		l.param.Slice(l2)[i] -= t2
	}
}

func (l *lexvec) observe(trained chan struct{}, clk *clock.Clock) {
	var cnt int
	for range trained {
		cnt++
		if cnt%l.opts.BatchSize == 0 {
			lower := l.opts.Initlr * l.opts.Theta
			if l.currentlr < lower {
				l.currentlr = lower
			} else {
				l.currentlr = l.opts.Initlr * (1.0 - float64(cnt)/float64(l.corpus.Len()))
			}
			l.verbose.Do(func() {
				fmt.Printf("trained %d words %v\r", cnt, clk.AllElapsed())
			})
		}
	}
	l.verbose.Do(func() {
		fmt.Printf("trained %d words %v\r\n", cnt, clk.AllElapsed())
	})
}

func (l *lexvec) Save(f io.Writer, typ save.VectorType) error {
	writer := bufio.NewWriter(f)
	defer writer.Flush()

	dic := l.corpus.Dictionary()

	var buf bytes.Buffer
	clk := clock.New()
	for i := 0; i < dic.Len(); i++ {
		word, _ := dic.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < l.opts.Dim; j++ {
			var v float64
			switch {
			case typ == save.Aggregated:
				v = l.param.Slice(i)[j] + l.param.Slice(i)[j]
			case typ == save.Single:
				v = l.param.Slice(i)[j]
			default:
				return save.InvalidVectorTypeError(typ)
			}
			fmt.Fprintf(&buf, "%f ", v)
		}
		fmt.Fprintln(&buf)
		l.verbose.Do(func() {
			fmt.Printf("save %d words %v\r", i, clk.AllElapsed())
		})
	}
	writer.WriteString(fmt.Sprintf("%v", buf.String()))
	l.verbose.Do(func() {
		fmt.Printf("save %d words %v\r\n", dic.Len(), clk.AllElapsed())
	})
	return nil
}
