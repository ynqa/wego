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

// Neighbor stores the word with cosine similarity value on the target.
type Neighbor struct {
	word       string
	similarity float64
}

// Neighbors is the list of Sim.
type Neighbors []Neighbor

func (n Neighbors) Len() int           { return len(n) }
func (n Neighbors) Less(i, j int) bool { return n[i].similarity < n[j].similarity }
func (n Neighbors) Swap(i, j int)      { n[i], n[j] = n[j], n[i] }
