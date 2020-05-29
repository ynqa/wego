package subsample

import (
	"math"
	"math/rand"

	"github.com/ynqa/wego/pkg/corpus/dictionary"
)

type Subsampler struct {
	samples []float64
}

func New(
	dic *dictionary.Dictionary,
	threshold float64,
) *Subsampler {
	samples := make([]float64, dic.Len())
	for i := 0; i < dic.Len(); i++ {
		z := 1. - math.Sqrt(threshold/float64(dic.IDFreq(i)))
		if z < 0 {
			z = 0
		}
		samples[i] = z
	}
	return &Subsampler{
		samples: samples,
	}
}

func (s *Subsampler) Trial(id int) bool {
	bernoulliTrial := rand.Float64()
	var ok bool
	if s.samples[id] > bernoulliTrial {
		ok = true
	}
	return ok
}
