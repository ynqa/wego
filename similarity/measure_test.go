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
	"testing"
)

func NewDummyMeasures() Measures {
	ms := make(Measures, 0)
	ms = append(ms, Measure{
		word:       "Cupcake",
		similarity: 0.1,
	})
	ms = append(ms, Measure{
		word:       "Donut",
		similarity: 0.2,
	})
	ms = append(ms, Measure{
		word:       "Eclair",
		similarity: 0.3,
	})
	ms = append(ms, Measure{
		word:       "Froyo",
		similarity: 0.4,
	})
	ms = append(ms, Measure{
		word:       "Gingerbread",
		similarity: 0.5,
	})
	return ms
}

func TestMeasuresLen(t *testing.T) {
	measures := NewDummyMeasures()

	if measures.Len() != 5 {
		t.Errorf("Expected len=5: %v", measures.Len())
	}
}

func TestMeasuresLess(t *testing.T) {
	measures := NewDummyMeasures()

	if !measures.Less(0, 3) {
		t.Errorf("Expected less(0, 3)=true: measures[0].similarity %v vs. measures[3].similarity %v",
			measures[0].similarity, measures[3].similarity)
	}
}

func TestMeasuresSwap(t *testing.T) {
	measures := NewDummyMeasures()
	m0 := measures[0]
	m3 := measures[3]

	measures.Swap(0, 3)

	if measures[0] != m3 {
		t.Errorf("Expected to equal %v to %v", measures[0], m3)
	}

	if measures[3] != m0 {
		t.Errorf("Expected to equal %v to %v", measures[3], m0)
	}
}
