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

package distance

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/olekukonko/tablewriter"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Estimator stores the elements for cosine similarity.
type Estimator struct {
	target string
	rank   int
	dense  map[string]*tensor.Dense
}

// NewEstimator creates *SimilarityEstimator
func NewEstimator(target string, rank int) *Estimator {
	return &Estimator{
		target: target,
		rank:   rank,
		dense:  make(map[string]*tensor.Dense),
	}
}

// Estimate estimates the similarity for target word.
func (e *Estimator) Estimate(f io.ReadCloser) error {
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, " ") {
			continue
		}
		word, vec, err := parse(line)
		if err != nil {
			return err
		}

		e.dense[word] = vec
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.New("Unable to complete scanning")
	}

	return nil
}

// Describe shows the similar words list for target word.
func (e *Estimator) Describe() error {
	// TODO: Save to the file also.
	return e.stdout()
}

func (e *Estimator) stdout() error {
	tvec, ok := e.dense[e.target]
	if !ok {
		return fmt.Errorf("%v is not found", e.target)
	}

	tvecNorm, err := norm(tvec)

	if err != nil {
		return err
	}

	res := make(Measures, len(e.dense))

	for word, vec := range e.dense {
		if word == e.target {
			continue
		}
		vecNorm, err := norm(vec)

		if err != nil {
			return err
		}

		sim, err := cosine(tvec, vec, tvecNorm, vecNorm)

		if err != nil {
			return err
		}

		res = append(res, Measure{
			word:       word,
			similarity: sim,
		})
	}

	sort.Sort(sort.Reverse(res))

	table := make([][]string, e.rank)
	for r := 0; r < e.rank; r++ {
		table[r] = []string{
			fmt.Sprintf("%d", r+1),
			res[r].word,
			fmt.Sprintf("%f", res[r].similarity),
		}
	}

	tw := tablewriter.NewWriter(os.Stdout)
	tw.SetHeader([]string{"Rank", "Word", "Cosine"})
	tw.SetBorder(false)
	tw.AppendBulk(table)
	tw.Render()
	return nil
}

func parse(line string) (string, *tensor.Dense, error) {
	sep := strings.Fields(line)
	word := sep[0]
	v := sep[1:]
	vec := tensor.NewDense(tensor.Float64, tensor.Shape{len(v)})
	dat := vec.Data().([]float64)
	for k, elem := range v {
		val, err := strconv.ParseFloat(elem, 64)
		if err != nil {
			return "", nil, err
		}
		dat[k] = val
	}
	return word, vec, nil
}

func norm(d *tensor.Dense) (float64, error) {
	ifNorm, err := d.Norm(tensor.UnorderedNorm())

	if err != nil {
		return 0, errors.Wrap(err, "Norm failed")
	}

	n := ifNorm.ScalarValue().(float64)
	return n, nil
}

func cosine(d1, d2 *tensor.Dense, d1Norm, d2Norm float64) (float64, error) {
	inner, err := tensor.Inner(d1, d2)

	if err != nil {
		return 0, errors.Wrap(err, "Inner failed")
	}

	cos := inner.(float64) / (d1Norm * d2Norm)
	return cos, nil
}
