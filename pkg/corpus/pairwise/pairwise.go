package pairwise

import (
	"math"

	"github.com/pkg/errors"

	"github.com/ynqa/wego/pkg/corpus/pairwise/encode"
)

type Pairwise struct {
	opts Options

	colloc map[uint64]float64
}

func New(opts Options) *Pairwise {
	return &Pairwise{
		opts: opts,

		colloc: make(map[uint64]float64),
	}
}

func (p *Pairwise) Colloc() map[uint64]float64 {
	return p.colloc
}

func (p *Pairwise) Add(left, right int) error {
	enc := encode.EncodeBigram(uint64(left), uint64(right))
	var val float64
	switch p.opts.CountType {
	case Increment:
		val = 1
	case Distance:
		div := left - right
		if div == 0 {
			return errors.Errorf("Divide by zero on counting co-occurrence")
		}
		val = 1. / math.Abs(float64(div))
	default:
		return invalidCountTypeError(p.opts.CountType)
	}
	p.colloc[enc] += val
	return nil
}
