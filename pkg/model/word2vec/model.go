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
	ch := make(chan []float64, opts.Goroutines)
	for i := 0; i < opts.Goroutines; i++ {
		ch <- make([]float64, opts.Dim)
	}
	return &skipGram{
		ch:     ch,
		window: opts.Window,
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

type cbowToken struct {
	agg []float64
	tmp []float64
}

type cbow struct {
	ch     chan cbowToken
	window int
}

func newCbow(opts Options) mod {
	ch := make(chan cbowToken, opts.Goroutines)
	for i := 0; i < opts.Goroutines; i++ {
		ch <- cbowToken{
			agg: make([]float64, opts.Dim),
			tmp: make([]float64, opts.Dim),
		}
	}
	return &cbow{
		ch:     ch,
		window: opts.Window,
	}
}

func (mod *cbow) trainOne(
	doc []int,
	pos int,
	lr float64,
	param *matrix.Matrix,
	optimizer optimizer,
) {
	token := <-mod.ch
	agg, tmp := token.agg, token.tmp
	defer func() {
		token := cbowToken{agg, tmp}
		mod.ch <- token
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
