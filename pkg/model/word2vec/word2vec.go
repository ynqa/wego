package word2vec

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
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/model/modelutil"
	"github.com/ynqa/wego/pkg/model/modelutil/matrix"
	"github.com/ynqa/wego/pkg/model/modelutil/save"
	"github.com/ynqa/wego/pkg/model/modelutil/subsample"
	"github.com/ynqa/wego/pkg/verbose"
)

type word2vec struct {
	opts Options

	corpus *corpus.Corpus

	param      *matrix.Matrix
	subsampler *subsample.Subsampler
	currentlr  float64
	mod        mod
	optimizer  optimizer

	verbose *verbose.Verbose
}

func New(opts ...ModelOption) (model.Model, error) {
	options := Options{
		CorpusOptions: corpus.DefaultOptions(),
		ModelOptions:  model.DefaultOptions(),

		MaxDepth:           defaultMaxDepth,
		ModelType:          defaultModelType,
		NegativeSampleSize: defaultNegativeSampleSize,
		OptimizerType:      defaultOptimizerType,
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
	return &word2vec{
		opts: opts,

		corpus: corpus.New(opts.CorpusOptions, v),

		currentlr: opts.ModelOptions.Initlr,

		verbose: v,
	}, nil
}

func (w *word2vec) preTrain(r io.Reader) error {
	if err := w.corpus.Build(r); err != nil {
		return err
	}

	dic, dim := w.corpus.Dictionary(), w.opts.ModelOptions.Dim

	w.param = matrix.New(
		dic.Len(),
		dim,
		func(vec []float64) {
			for i := 0; i < dim; i++ {
				vec[i] = (rand.Float64() - 0.5) / float64(dim)
			}
		},
	)

	w.subsampler = subsample.New(dic, w.opts.SubsampleThreshold)

	switch w.opts.ModelType {
	case SkipGram:
		w.mod = newSkipGram(w.opts)
	case Cbow:
		w.mod = newCbow(w.opts)
	default:
		return invalidModelTypeError(w.opts.ModelType)
	}

	switch w.opts.OptimizerType {
	case NegativeSampling:
		w.optimizer = newNegativeSampling(
			w.corpus.Dictionary(),
			w.opts,
		)
	case HierarchicalSoftmax:
		w.optimizer = newHierarchicalSoftmax(
			w.corpus.Dictionary(),
			w.opts,
		)
	default:
		return invalidOptimizerTypeError(w.opts.OptimizerType)
	}
	return nil
}

func (w *word2vec) Train(r io.Reader) error {
	if err := w.preTrain(r); err != nil {
		return err
	}

	doc := w.corpus.Doc()
	indexPerThread := modelutil.IndexPerThread(
		w.opts.ModelOptions.ThreadSize,
		len(doc),
	)

	for i := 1; i <= w.opts.ModelOptions.Iter; i++ {
		trained, clk := make(chan struct{}), clock.New()
		go w.observe(trained, clk)

		sem := semaphore.NewWeighted(int64(w.opts.ModelOptions.ThreadSize))
		wg := &sync.WaitGroup{}

		for i := 0; i < w.opts.ModelOptions.ThreadSize; i++ {
			wg.Add(1)
			s, e := indexPerThread[i], indexPerThread[i+1]
			go w.trainPerThread(doc[s:e], trained, sem, wg)
		}

		wg.Wait()
		close(trained)
	}
	return nil
}

func (w *word2vec) trainPerThread(
	doc []int,
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
		if w.subsampler.Trial(id) {
			w.mod.trainOne(doc, pos, w.currentlr, w.param, w.optimizer)
		}
		trained <- struct{}{}
	}

	return nil
}

func (w *word2vec) observe(trained chan struct{}, clk *clock.Clock) {
	var cnt int
	for range trained {
		cnt++
		if cnt%w.opts.ModelOptions.BatchSize == 0 {
			lower := w.opts.ModelOptions.Initlr * w.opts.Theta
			if w.currentlr < lower {
				w.currentlr = lower
			} else {
				w.currentlr = w.opts.ModelOptions.Initlr * (1.0 - float64(cnt)/float64(w.corpus.Len()))
			}
			w.verbose.Do(func() {
				fmt.Printf("trained %d words %v\r", cnt, clk.AllElapsed())
			})
		}
	}
	w.verbose.Do(func() {
		fmt.Printf("trained %d words %v\r\n", cnt, clk.AllElapsed())
	})
}

func (w *word2vec) Save(f io.Writer, typ save.VectorType) error {
	writer := bufio.NewWriter(f)
	defer writer.Flush()

	dic := w.corpus.Dictionary()
	var ctx *matrix.Matrix
	ng, ok := w.optimizer.(*negativeSampling)
	if ok {
		ctx = ng.ctx
	}

	var buf bytes.Buffer
	clk := clock.New()
	for i := 0; i < dic.Len(); i++ {
		word, _ := dic.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < w.opts.ModelOptions.Dim; j++ {
			var v float64
			switch {
			case typ == save.AggregatedVector && ctx.Row() > i:
				v = w.param.Slice(i)[j] + ctx.Slice(i)[j]
			case typ == save.SingleVector:
				v = w.param.Slice(i)[j]
			default:
				return save.InvalidVectorTypeError(typ)
			}
			fmt.Fprintf(&buf, "%f ", v)
		}
		fmt.Fprintln(&buf)
		w.verbose.Do(func() {
			fmt.Printf("save %d words %v\r", i, clk.AllElapsed())
		})
	}
	writer.WriteString(fmt.Sprintf("%v", buf.String()))
	w.verbose.Do(func() {
		fmt.Printf("save %d words %v\r\n", dic.Len(), clk.AllElapsed())
	})
	return nil
}
