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
