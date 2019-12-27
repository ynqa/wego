package word2vec

import (
	"github.com/ynqa/wego/pkg/model/modelutil"
	"github.com/ynqa/wego/pkg/model/modelutil/matrix"
)

type mod interface {
	trainOne(
		doc []int,
		pos int,
		lr float64,
		param *matrix.Matrix,
		optimizer optimizer,
	)
}

type skipGram struct {
	ch     chan []float64
	window int
}

func newSkipGram(opts Options) mod {
	ch := make(chan []float64, opts.ModelOptions.ThreadSize)
	for i := 0; i < opts.ModelOptions.ThreadSize; i++ {
		ch <- make([]float64, opts.ModelOptions.Dim)
	}
	return &skipGram{
		ch:     ch,
		window: opts.ModelOptions.Window,
	}
}

func (mod *skipGram) trainOne(
	doc []int,
	pos int,
	lr float64,
	param *matrix.Matrix,
	optimizer optimizer,
) {
	tmp := <-mod.ch
	defer func() {
		mod.ch <- tmp
	}()
	del := modelutil.NextRandom(mod.window)
	for a := del; a < mod.window*2+1-del; a++ {
		if a == mod.window {
			continue
		}
		c := pos - mod.window + a
		if c < 0 || c >= len(doc) {
			continue
		}
		for i := 0; i < len(tmp); i++ {
			tmp[i] = 0
		}
		ctxID := doc[c]
		ctx := param.Slice(ctxID)
		optimizer.optim(doc[pos], lr, ctx, tmp)
		for i := 0; i < len(ctx); i++ {
			ctx[i] += tmp[i]
		}
	}
}

type cbow struct {
	ch     chan []float64
	window int
}

func newCbow(opts Options) mod {
	ch := make(chan []float64, opts.ModelOptions.ThreadSize*2)
	for i := 0; i < opts.ModelOptions.ThreadSize; i++ {
		ch <- make([]float64, opts.ModelOptions.Dim)
	}
	return &cbow{
		ch:     ch,
		window: opts.ModelOptions.Window,
	}
}

func (mod *cbow) trainOne(
	doc []int,
	pos int,
	lr float64,
	param *matrix.Matrix,
	optimizer optimizer,
) {
	agg, tmp := <-mod.ch, <-mod.ch
	defer func() {
		mod.ch <- agg
		mod.ch <- tmp
	}()
	for i := 0; i < len(agg); i++ {
		agg[i], tmp[i] = 0, 0
	}
	mod.dowith(doc, pos, param, agg, tmp, mod.aggregate)
	optimizer.optim(doc[pos], lr, agg, tmp)
	mod.dowith(doc, pos, param, agg, tmp, mod.update)
}

func (mod *cbow) dowith(
	doc []int,
	pos int,
	param *matrix.Matrix,
	agg, tmp []float64,
	fn func(ctx, agg, tmp []float64),
) {
	del := modelutil.NextRandom(mod.window)
	for a := del; a < mod.window*2+1-del; a++ {
		if a == mod.window {
			continue
		}
		c := pos - mod.window + a
		if c < 0 || c >= len(doc) {
			continue
		}
		ctxID := doc[c]
		ctx := param.Slice(ctxID)
		fn(ctx, agg, tmp)
	}
}

func (c *cbow) aggregate(ctx, agg, _ []float64) {
	for i := 0; i < len(ctx); i++ {
		agg[i] += ctx[i]
	}
}

func (c *cbow) update(ctx, _, tmp []float64) {
	for i := 0; i < len(ctx); i++ {
		ctx[i] += tmp[i]
	}
}
