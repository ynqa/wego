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
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/olekukonko/tablewriter"
	"github.com/ynqa/word-embedding/utils/fileio"
	"github.com/ynqa/word-embedding/utils/vector"
)

// VectorMap is the map composed of <word, Vector>.
type VectorMap map[string]vector.Vector

var vectorMap VectorMap

// Sim stores word, and cosine similarity value against target.
type Sim struct {
	word   string
	cosine float64
}

// SimList is the list of Sim.
type SimList []Sim

func (p SimList) Len() int           { return len(p) }
func (p SimList) Less(i, j int) bool { return p[i].cosine < p[j].cosine }
func (p SimList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func init() {
	vectorMap = make(VectorMap)
}

// Load reads trained vector.
func Load(inputFile string) error {
	fileio.LoadVector(inputFile, func(lines []string) {
		for _, s := range lines {
			word, vec := parse(s)
			vectorMap[word] = vec
		}
	})
	return nil
}

func parse(line string) (string, vector.Vector) {
	sep := strings.Fields(line)
	word := sep[0]
	v := sep[1:]
	vec := vector.NewVector(len(v))
	for k, pair := range v {
		val, err := strconv.ParseFloat(strings.Split(pair, ":")[1], 64)
		if err != nil {
			return "", nil
		}
		vec[k] = val
	}
	return word, vec
}

// Describe shows the similar list against target.
func Describe(target string, rank int) error {
	targetVector, existed := vectorMap[target]
	if !existed {
		return fmt.Errorf("%v is not found", target)
	}

	res := make(SimList, 0)
	for word, vec := range vectorMap {
		if word == target {
			continue
		}
		res = append(res, Sim{
			word:   word,
			cosine: targetVector.Cosine(vec),
		})
	}

	sort.Sort(sort.Reverse(res))

	table := make([][]string, 100)
	for r := 0; r < rank; r++ {
		table[r] = []string{
			fmt.Sprintf("%d", r+1),
			res[r].word,
			fmt.Sprintf("%f", res[r].cosine),
		}
	}

	tw := tablewriter.NewWriter(os.Stdout)
	tw.SetHeader([]string{"Rank", "Word", "Cosine"})
	tw.SetBorder(false)
	tw.AppendBulk(table)
	tw.Render()
	return nil
}
