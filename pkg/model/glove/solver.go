package glove

import (
	"math"

	"github.com/ynqa/wego/pkg/corpus/dictionary"
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/model/modelutil/matrix"
)

type solver interface {
	trainOne(l1, l2 int, param *matrix.Matrix, f, coef float64)
}

type stochastic struct {
	initlr float64
}

func newStochastic(opts model.Options) solver {
	return &stochastic{
		initlr: opts.Initlr,
	}
}

func (sol *stochastic) trainOne(l1, l2 int, param *matrix.Matrix, f, coef float64) {
	v1, v2 := param.Slice(l1), param.Slice(l2)
	dim, diff := len(v1)-1, 0.
	for i := 0; i < dim; i++ {
		diff += v1[i] * v2[i]
	}
	diff += v1[dim] + v2[dim] - f
	diff *= coef * sol.initlr
	for i := 0; i < dim; i++ {
		t1, t2 := diff*v2[i], diff*v1[i]
		v1[i] -= t1
		v2[i] -= t2
	}
	v1[dim] -= diff
	v2[dim] -= diff
}

type adaGrad struct {
	initlr float64
	gradsq *matrix.Matrix
}

func newAdaGrad(dic *dictionary.Dictionary, opts model.Options) solver {
	dimAndBias := opts.Dim + 1
	return &adaGrad{
		initlr: opts.Initlr,
		gradsq: matrix.New(
			dic.Len()*2,
			dimAndBias,
			func(vec []float64) {
				for i := 0; i < dimAndBias; i++ {
					vec[i] = 1.
				}
			},
		),
	}
}

func (sol *adaGrad) trainOne(l1, l2 int, param *matrix.Matrix, f, coef float64) {
	v1, v2 := param.Slice(l1), param.Slice(l2)
	g1, g2 := sol.gradsq.Slice(l1), sol.gradsq.Slice(l2)
	dim, diff := len(v1)-1, 0.
	for i := 0; i < dim; i++ {
		diff += v1[i] * v2[i]
	}
	diff += v1[dim] + v2[dim] - f
	diff *= coef * sol.initlr
	for i := 0; i < dim; i++ {
		t1, t2 := diff*v2[i], diff*v1[i]
		g1[i] += t1 * t1
		g2[i] += t2 * t2
		t1 /= math.Sqrt(g1[i])
		t2 /= math.Sqrt(g2[i])
		v1[i] -= t1
		v2[i] -= t2
	}
	v1[dim] -= diff / math.Sqrt(g1[dim])
	v2[dim] -= diff / math.Sqrt(g2[dim])
	diff *= diff
	g1[dim] += diff
	g2[dim] += diff
}
