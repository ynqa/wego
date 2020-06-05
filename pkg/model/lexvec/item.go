package lexvec

import (
	"fmt"
	"math"

	"github.com/pkg/errors"
	"github.com/ynqa/wego/pkg/clock"
	"github.com/ynqa/wego/pkg/corpus/pairwise"
	"github.com/ynqa/wego/pkg/corpus/pairwise/encode"
)

func (l *lexvec) makeItems(pairwise *pairwise.Pairwise) (map[uint64]float64, error) {
	pm := pairwise.PairMap()
	res, idx, clk := make(map[uint64]float64), 0, clock.New()
	logTotalFreq := math.Log(math.Pow(float64(l.corpus.Len()), l.opts.Smooth))
	for enc, f := range pm {
		u1, u2 := encode.DecodeBigram(enc)
		l1, l2 := int(u1), int(u2)
		v, err := l.calculateRelation(
			l.opts.RelationType,
			l1, l2,
			f, logTotalFreq,
		)
		if err != nil {
			return nil, err
		}
		res[enc] = v
		idx++
		l.verbose.Do(func() {
			if idx%100000 == 0 {
				fmt.Printf("build %d items %v\r", idx, clk.AllElapsed())
			}
		})
	}
	l.verbose.Do(func() {
		fmt.Printf("build %d items %v\r\n", idx, clk.AllElapsed())
	})
	return res, nil
}

func (l *lexvec) calculateRelation(
	typ RelationType,
	l1, l2 int,
	co, logTotalFreq float64,
) (float64, error) {
	dic := l.corpus.Dictionary()
	switch typ {
	case PPMI:
		if co == 0 {
			return 0, nil
		}
		// TODO: avoid log for l1, l2 every time
		ppmi := math.Log(co) - math.Log(float64(dic.IDFreq(l1))) - math.Log(math.Pow(float64(dic.IDFreq(l2)), l.opts.Smooth)) + logTotalFreq
		if ppmi < 0 {
			ppmi = 0
		}
		return ppmi, nil
	case PMI:
		if co == 0 {
			return 1, nil
		}
		pmi := math.Log(co) - math.Log(float64(dic.IDFreq(l1))) - math.Log(math.Pow(float64(dic.IDFreq(l2)), l.opts.Smooth)) + logTotalFreq
		return pmi, nil
	case Collocation:
		return co, nil
	case LogCollocation:
		return math.Log(co), nil
	default:
		return 0, errors.Errorf("invalid measure type")
	}
}
