// Copyright © 2019 Makoto Ito
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

// CountModelCorpus stores corpus and co-occurrence values between words.
type CountModelCorpus struct {
	*core
}

// Pair stores co-occurrence information.
type Pair struct {
	// L1 and L2 store index number for two co-occurrence words.
	L1, L2 int
	// F stores the measures of co-occurrence, such as PMI.
	F float64
	// Coefficient stores a coefficient for weighted matrix factorization.
	Coefficient float64
}

// PairMap stores co-occurrences.
type PairMap map[uint64]float64

// RelationType is a list of types for strength relations between co-occurrence words.
type RelationType int

const (
	PPMI RelationType = iota
	PMI
	CO
	LOGCO
)

// String describes relation type name.
func (r RelationType) String() string {
	switch r {
	case PPMI:
		return "ppmi"
	case PMI:
		return "pmi"
	case CO:
		return "co"
	case LOGCO:
		return "logco"
	default:
		return "unknown"
	}
}

func (c *CountModelCorpus) relationValue(typ RelationType, l1, l2 int, co, logTotalFreq, smooth float64) (float64, error) {
	switch typ {
	case PPMI:
		if co == 0 {
			return 0, nil
		}
		// TODO: avoid log for l1, l2 every time
		ppmi := math.Log(co) - math.Log(float64(c.IDFreq(l1))) - math.Log(math.Pow(float64(c.IDFreq(l2)), smooth)) + logTotalFreq
		if ppmi < 0 {
			ppmi = 0
		}
		return ppmi, nil
	case PMI:
		if co == 0 {
			return 1, nil
		}
		pmi := math.Log(co) - math.Log(float64(c.IDFreq(l1))) - math.Log(math.Pow(float64(c.IDFreq(l2)), smooth)) + logTotalFreq
		return pmi, nil
	case CO:
		return co, nil
	case LOGCO:
		return math.Log(co), nil
	default:
		return 0, errors.Errorf("Invalid measure type")
	}
}

// CountType is a list of types to count co-occurences.
type CountType int

const (
	INCREMENT CountType = iota
	// DISTANCE weights values for co-occurrence times.
	DISTANCE
)

func countValue(typ CountType, left, right int) (float64, error) {
	switch typ {
	case INCREMENT:
		return 1., nil
	case DISTANCE:
		div := left - right
		if div == 0 {
			return 0, errors.Errorf("Divide by zero on counting co-occurrence")
		}
		return 1. / math.Abs(float64(div)), nil
	default:
		return 0, errors.Errorf("Invalid count type")
	}
}

// NewCountModelCorpus creates *CountModelCorpus.
func NewCountModelCorpus(f io.ReadCloser, toLower bool, minCount int) (*CountModelCorpus, error) {
	c := &CountModelCorpus{
		core: newCore(),
	}
	if err := c.parse(f, toLower, minCount); err != nil {
		return nil, errors.Wrap(err, "Unable to generate *CountModelCorpus")
	}
	return c, nil
}

func (c *CountModelCorpus) cooccurrence(window int, typ CountType, verbose bool) (PairMap, error) {
	documentSize := len(c.document)

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
			f, err := countValue(typ, i, j)
			if err != nil {
				return nil, errors.Wrap(err, "Failed to count co-occurrence between words")
			}
			cooccurrence[co.EncodeBigram(uint64(c.document[i]), uint64(c.document[j]))] += f
			cooccurrence[co.EncodeBigram(uint64(c.document[j]), uint64(c.document[i]))] += f
		}
		if verbose {
			progress.Increment()
		}
	}
	return cooccurrence, nil
}

func (c *CountModelCorpus) PairsIntoLexvec(window int, relationType RelationType, smooth float64, verbose bool) (map[uint64]float64, error) {
	cooccurrence, err := c.cooccurrence(window, INCREMENT, verbose)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create Pairs for Lexvec")
	}
	cooccurrenceSize := len(cooccurrence)

	var progress *pb.ProgressBar
	if verbose {
		fmt.Println("Scan cooccurrences for pairs")
		progress = pb.New(cooccurrenceSize).SetWidth(80)
		defer progress.Finish()
		progress.Start()
	}

	logTotalFreq := math.Log(math.Pow(float64(c.TotalFreq()), smooth))
	for p, f := range cooccurrence {
		ul1, ul2 := co.DecodeBigram(p)
		v, err := c.relationValue(relationType, int(ul1), int(ul2), f, logTotalFreq, smooth)
		if err != nil {
			return nil, errors.Wrap(err, "Failed to calculate relation value")
		}
		cooccurrence[p] = v
		if verbose {
			progress.Increment()
		}
	}
	return cooccurrence, nil
}

func (c *CountModelCorpus) PairsIntoGlove(window int, xmax int, alpha float64, verbose bool) ([]Pair, error) {
	cooccurrence, err := c.cooccurrence(window, DISTANCE, verbose)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to create Pairs for GloVe")
	}
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
	return pairs, nil
}
