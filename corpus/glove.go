// Copyright © 2017 Makoto Ito
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

package corpus

import (
	"fmt"
	"io"
	"math"
	"math/rand"

	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/wego/corpus/co"
)

// GloveCorpus stores corpus.
type GloveCorpus struct {
	*core
}

// NewGloveCorpus creates *GloveCorpus.
func NewGloveCorpus(f io.ReadCloser, toLower bool, minCount int) (*GloveCorpus, error) {
	gloveCorpus := &GloveCorpus{
		core: newCore(),
	}
	if err := gloveCorpus.parse(f, toLower, minCount); err != nil {
		return nil, errors.Wrap(err, "Unable to generate *GloveCorpus")
	}
	return gloveCorpus, nil
}

type Pair struct {
	L1, L2         int
	F, Coefficient float64
}

func (gc *GloveCorpus) cooccurrence(window int, verbose bool) map[uint64]float64 {
	documentSize := len(gc.Document())

	var progress *pb.ProgressBar
	if verbose {
		fmt.Println("Scan corpus for cooccurrences")
		progress = pb.New(documentSize).SetWidth(80)
		defer progress.Finish()
		progress.Start()
	}
	cooccurrence := make(map[uint64]float64)
	for i := 0; i < documentSize; i++ {
		for j := i + 1; j <= i+window; j++ {
			if j >= documentSize {
				continue
			}
			f := 1. / math.Abs(float64(i-j))
			cooccurrence[co.EncodeBigram(uint64(gc.document[i]), uint64(gc.document[j]))] += f
			cooccurrence[co.EncodeBigram(uint64(gc.document[j]), uint64(gc.document[i]))] += f
		}
		if verbose {
			progress.Increment()
		}
	}
	return cooccurrence
}

func (gc *GloveCorpus) Pairs(window int, xmax int, alpha float64, verbose bool) []Pair {
	cooccurrence := gc.cooccurrence(window, verbose)
	pairSize := len(cooccurrence)
	pairs := make([]Pair, pairSize)
	shuffle := rand.Perm(pairSize)

	var progress *pb.ProgressBar
	if verbose {
		fmt.Println("Scan cooccurrences for pairs")
		progress = pb.New(pairSize).SetWidth(80)
		defer progress.Finish()
		progress.Start()
	}

	var i int
	for p, f := range cooccurrence {
		coefficient := 1.0
		if f < float64(xmax) {
			coefficient = math.Pow(f/float64(xmax), alpha)
		}

		ul1, ul2 := co.DecodeBigram(p)
		pairs[shuffle[i]] = Pair{
			L1:          int(ul1),
			L2:          int(ul2),
			F:           math.Log(f),
			Coefficient: coefficient,
		}
		i++
		if verbose {
			progress.Increment()
		}
	}
	return pairs
}
