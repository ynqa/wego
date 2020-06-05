package glove

import (
	"fmt"
	"math"

	"github.com/ynqa/wego/pkg/clock"
	"github.com/ynqa/wego/pkg/corpus/pairwise"
	"github.com/ynqa/wego/pkg/corpus/pairwise/encode"
)

type item struct {
	l1, l2 int
	f      float64
	coef   float64
}

func (g *glove) makeItems(pairwise *pairwise.Pairwise) []item {
	pm := pairwise.PairMap()
	res, idx, clk := make([]item, len(pm)), 0, clock.New()
	for enc, f := range pm {
		u1, u2 := encode.DecodeBigram(enc)
		l1, l2 := int(u1), int(u2)
		coef := 1.
		if f < float64(g.opts.Xmax) {
			coef = math.Pow(f/float64(g.opts.Xmax), g.opts.Alpha)
		}
		res[idx] = item{
			l1:   l1,
			l2:   l2,
			f:    math.Log(f),
			coef: coef,
		}
		idx++
		g.verbose.Do(func() {
			if idx%100000 == 0 {
				fmt.Printf("build %d items %v\r", idx, clk.AllElapsed())
			}
		})
	}
	g.verbose.Do(func() {
		fmt.Printf("build %d items %v\r\n", idx, clk.AllElapsed())
	})
	return res
}
