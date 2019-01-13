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

package search

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/olekukonko/tablewriter"
	"github.com/pkg/errors"
)

// Searcher stores the elements for cosine similarity.
type Searcher struct {
	target  string
	rank    int
	vectors map[string][]float64
}

// NewSearcher creates *Searcher
func NewSearcher(target string, rank int) *Searcher {
	return &Searcher{
		target:  target,
		rank:    rank,
		vectors: make(map[string][]float64),
	}
}

// Search searches similar words for target word.
func (s *Searcher) Search(f io.ReadCloser) error {
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

		s.vectors[word] = vec
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return errors.New("Unable to complete scanning")
	}

	return nil
}

// Describe shows the similar words list for target word.
func (s *Searcher) Describe() error {
	// TODO: Save to the file also.
	return s.stdout()
}

func (s *Searcher) stdout() error {
	targetVec, ok := s.vectors[s.target]
	if !ok {
		return fmt.Errorf("%v is not found", s.target)
	}
	targetNorm := norm(targetVec)
	res := make(Measures, len(s.vectors))

	for word, vec := range s.vectors {
		if word == s.target {
			continue
		}
		n := norm(vec)
		res = append(res, Measure{
			word:       word,
			similarity: cosine(targetVec, vec, targetNorm, n),
		})
	}

	sort.Sort(sort.Reverse(res))

	table := make([][]string, s.rank)
	for r := 0; r < s.rank; r++ {
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

func parse(line string) (string, []float64, error) {
	sep := strings.Fields(line)
	word := sep[0]
	elems := sep[1:]
	vec := make([]float64, len(elems))
	for k, elem := range elems {
		val, err := strconv.ParseFloat(elem, 64)
		if err != nil {
			return "", nil, err
		}
		vec[k] = val
	}
	return word, vec, nil
}

func norm(vec []float64) float64 {
	var n float64
	for _, v := range vec {
		n += math.Pow(v, 2)
	}
	return math.Sqrt(n)
}

func cosine(v1, v2 []float64, n1, n2 float64) float64 {
	var dot float64
	for i := range v1 {
		dot += v1[i] * v2[i]
	}
	return dot / n1 / n2
}
