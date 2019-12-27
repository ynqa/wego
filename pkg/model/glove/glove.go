package glove

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
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/model/modelutil"
	"github.com/ynqa/wego/pkg/model/modelutil/matrix"
	"github.com/ynqa/wego/pkg/model/modelutil/save"
	"github.com/ynqa/wego/pkg/model/subsample"
	"github.com/ynqa/wego/pkg/verbose"
)

type glove struct {
	opts Options

	corpus *corpus.Corpus

	param      *matrix.Matrix
	subsampler *subsample.Subsampler
	solver     solver

	verbose *verbose.Verbose
}

func New(opts ...ModelOption) (model.Model, error) {
	options := Options{
		CorpusOptions:   corpus.DefaultOptions(),
		PairwiseOptions: pairwise.DefaultOptions(),
		ModelOptions:    model.DefaultOptions(),

		Alpha:      defaultAlpha,
		SolverType: defaultSolverType,
		Xmax:       defaultXmax,
	}

	for _, fn := range opts {
		fn(&options)
	}

	return NewForOptions(options)
}

func NewForOptions(opts Options) (model.Model, error) {
	// TODO: validate Options
	v := verbose.New(opts.ModelOptions.Verbose)
	return &glove{
		opts: opts,

		corpus: corpus.New(opts.CorpusOptions, v),

		verbose: v,
	}, nil
}

func (g *glove) preTrain(r io.Reader) error {
	if err := g.corpus.BuildWithPairwise(
		r,
		g.opts.PairwiseOptions,
		g.opts.ModelOptions.Window,
	); err != nil {
		return err
	}

	dic, dim := g.corpus.Dictionary(), g.opts.ModelOptions.Dim

	g.param = matrix.New(
		dic.Len()*2,
		(dim + 1),
		func(vec []float64) {
			for i := 0; i < dim+1; i++ {
				vec[i] = rand.Float64() / float64(dim)
			}
		},
	)

	g.subsampler = subsample.New(dic, g.opts.SubsampleThreshold)

	switch g.opts.SolverType {
	case Stochastic:
		g.solver = newStochastic(g.opts.ModelOptions)
	case AdaGrad:
		g.solver = newAdaGrad(dic, g.opts.ModelOptions)
	default:
		return invalidSolverTypeError(g.opts.SolverType)
	}
	return nil
}

func (g *glove) Train(r io.Reader) error {
	if err := g.preTrain(r); err != nil {
		return err
	}

	items := g.preCalculateItems(g.corpus.Pairwise())
	itemSize := len(items)
	indexPerThread := modelutil.IndexPerThread(
		g.opts.ModelOptions.ThreadSize,
		itemSize,
	)

	for i := 0; i < g.opts.ModelOptions.Iter; i++ {
		trained, clk := make(chan struct{}), clock.New()
		go g.observe(trained, clk)

		sem := semaphore.NewWeighted(int64(g.opts.ModelOptions.ThreadSize))
		wg := &sync.WaitGroup{}

		for i := 0; i < g.opts.ModelOptions.ThreadSize; i++ {
			wg.Add(1)
			s, e := indexPerThread[i], indexPerThread[i+1]
			go g.trainPerThread(items[s:e], trained, sem, wg)
		}

		wg.Wait()
		close(trained)
	}
	return nil
}

func (g *glove) trainPerThread(
	items []item,
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

	dic := g.corpus.Dictionary()
	for _, item := range items {
		if g.subsampler.Trial(item.l1) &&
			g.subsampler.Trial(item.l2) &&
			dic.IDFreq(item.l1) > g.opts.ModelOptions.MinCount &&
			dic.IDFreq(item.l2) > g.opts.ModelOptions.MinCount {
			g.solver.trainOne(item.l1, item.l2+dic.Len(), g.param, item.f, item.coef)
			g.solver.trainOne(item.l1+dic.Len(), item.l2, g.param, item.f, item.coef)
		}
		trained <- struct{}{}
	}

	return nil
}

func (g *glove) observe(trained chan struct{}, clk *clock.Clock) {
	var cnt int
	for range trained {
		g.verbose.Do(func() {
			cnt++
			if cnt%g.opts.ModelOptions.BatchSize == 0 {
				fmt.Printf("trained %d items %v\r", cnt, clk.AllElapsed())
			}
		})
	}
	g.verbose.Do(func() {
		fmt.Printf("trained %d items %v\r\n", cnt, clk.AllElapsed())
	})
}

func (g *glove) Save(f io.Writer, typ save.VectorType) error {
	writer := bufio.NewWriter(f)
	defer writer.Flush()

	dic := g.corpus.Dictionary()

	var buf bytes.Buffer
	clk := clock.New()
	for i := 0; i < dic.Len(); i++ {
		word, _ := dic.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < g.opts.ModelOptions.Dim; j++ {
			var v float64
			switch {
			case typ == save.AggregatedVector:
				v = g.param.Slice(i)[j] + g.param.Slice(i + dic.Len())[j]
			case typ == save.SingleVector:
				v = g.param.Slice(i)[j]
			default:
				return save.InvalidVectorTypeError(typ)
			}
			fmt.Fprintf(&buf, "%f ", v)
		}
		fmt.Fprintln(&buf)
		g.verbose.Do(func() {
			fmt.Printf("save %d words %v\r", i, clk.AllElapsed())
		})
	}
	writer.WriteString(fmt.Sprintf("%v", buf.String()))
	g.verbose.Do(func() {
		fmt.Printf("save %d words %v\r\n", dic.Len(), clk.AllElapsed())
	})
	return nil
}
