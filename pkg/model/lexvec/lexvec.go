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
	"github.com/ynqa/wego/pkg/corpus/pairwise"
	"github.com/ynqa/wego/pkg/corpus/pairwise/encode"
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/model/modelutil"
	"github.com/ynqa/wego/pkg/model/modelutil/matrix"
	"github.com/ynqa/wego/pkg/model/modelutil/save"
	"github.com/ynqa/wego/pkg/model/subsample"
	"github.com/ynqa/wego/pkg/verbose"
)

type lexvec struct {
	opts Options

	corpus *corpus.Corpus

	param      *matrix.Matrix
	subsampler *subsample.Subsampler
	currentlr  float64

	verbose *verbose.Verbose
}

func New(opts ...ModelOption) (model.Model, error) {
	options := Options{
		CorpusOptions: corpus.DefaultOptions(),
		ModelOptions:  model.DefaultOptions(),

		NegativeSampleSize: defaultNegativeSampleSize,
		RelationType:       defaultRelationType,
		Smooth:             defaultSmooth,
		SubsampleThreshold: defaultSubsampleThreshold,
		Theta:              defaultTheta,
	}

	for _, fn := range opts {
		fn(&options)
	}

	return NewForOptions(options)
}

func NewForOptions(opts Options) (model.Model, error) {
	// TODO: validate Options
	v := verbose.New(opts.ModelOptions.Verbose)
	return &lexvec{
		opts: opts,

		corpus: corpus.New(opts.CorpusOptions, v),

		currentlr: opts.ModelOptions.Initlr,

		verbose: v,
	}, nil
}

func (l *lexvec) preTrain(r io.Reader) error {
	if err := l.corpus.BuildWithPairwise(
		r,
		pairwise.Options{
			CountType: pairwise.Increment,
		},
		l.opts.ModelOptions.Window,
	); err != nil {
		return err
	}

	dic, dim := l.corpus.Dictionary(), l.opts.ModelOptions.Dim

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

	items, err := l.preCalculateItems(l.corpus.Pairwise())
	if err != nil {
		return err
	}
	doc := l.corpus.Doc()
	indexPerThread := modelutil.IndexPerThread(
		l.opts.ModelOptions.ThreadSize,
		len(doc),
	)

	for i := 1; i <= l.opts.ModelOptions.Iter; i++ {
		trained, clk := make(chan struct{}), clock.New()
		go l.observe(trained, clk)

		sem := semaphore.NewWeighted(int64(l.opts.ModelOptions.ThreadSize))
		wg := &sync.WaitGroup{}

		for i := 0; i < l.opts.ModelOptions.ThreadSize; i++ {
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

	dic := l.corpus.Dictionary()
	for pos, id := range doc {
		if l.subsampler.Trial(id) && dic.IDFreq(id) > l.opts.ModelOptions.MinCount {
			l.trainOne(doc, pos, items)
		}
		trained <- struct{}{}
	}

	return nil
}

func (l *lexvec) trainOne(doc []int, pos int, items map[uint64]float64) {
	dic := l.corpus.Dictionary()
	del := modelutil.NextRandom(l.opts.ModelOptions.Window)
	for a := del; a < l.opts.ModelOptions.Window*2+1-del; a++ {
		if a == l.opts.ModelOptions.Window {
			continue
		}
		c := pos - l.opts.ModelOptions.Window + a
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
	for i := 0; i < l.opts.ModelOptions.Dim; i++ {
		diff += l.param.Slice(l1)[i] * l.param.Slice(l2)[i]
	}
	diff = (diff - f) * l.currentlr
	for i := 0; i < l.opts.ModelOptions.Dim; i++ {
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
		if cnt%l.opts.ModelOptions.BatchSize == 0 {
			lower := l.opts.ModelOptions.Initlr * l.opts.Theta
			if l.currentlr < lower {
				l.currentlr = lower
			} else {
				l.currentlr = l.opts.ModelOptions.Initlr * (1.0 - float64(cnt)/float64(l.corpus.Len()))
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
		for j := 0; j < l.opts.ModelOptions.Dim; j++ {
			var v float64
			switch {
			case typ == save.AggregatedVector:
				v = l.param.Slice(i)[j] + l.param.Slice(i)[j]
			case typ == save.SingleVector:
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
