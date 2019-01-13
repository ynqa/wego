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

// Measure stores the word with cosine similarity value on the target.
type Measure struct {
	word       string
	similarity float64
}

// Measures is the list of Sim.
type Measures []Measure

func (m Measures) Len() int           { return len(m) }
func (m Measures) Less(i, j int) bool { return m[i].similarity < m[j].similarity }
func (m Measures) Swap(i, j int)      { m[i], m[j] = m[j], m[i] }
