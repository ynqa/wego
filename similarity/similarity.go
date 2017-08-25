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

package similarity

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

	"github.com/ynqa/word-embedding/vector"
)

// Estimator stores the elements for cosine similarity.
type Estimator struct {
	target string
	rank   int
	tensor map[string]vector.Vector
}

// NewEstimator creates *SimilarityEstimator
func NewEstimator(target string, rank int) *Estimator {
	return &Estimator{
		target: target,
		rank:   rank,
		tensor: make(map[string]vector.Vector),
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

		e.tensor[word] = vec
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
	tvec, ok := e.tensor[e.target]
	if !ok {
		return fmt.Errorf("%v is not found", e.target)
	}

	res := make(Measures, len(e.tensor))

	for word, vec := range e.tensor {
		if word == e.target {
			continue
		}
		res = append(res, Measure{
			word:       word,
			similarity: tvec.Cosine(vec),
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

func parse(line string) (string, vector.Vector, error) {
	sep := strings.Fields(line)
	word := sep[0]
	v := sep[1:]
	vec := vector.NewVector(len(v))
	for k, pair := range v {
		val, err := strconv.ParseFloat(strings.Split(pair, ":")[1], 64)
		if err != nil {
			return "", nil, err
		}
		vec[k] = val
	}
	return word, vec, nil
}
