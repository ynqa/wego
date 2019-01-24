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

package search

import (
	"testing"
)

func TestDescribe(t *testing.T) {
	neighbors := Neighbors{
		Neighbor{
			word:       "a",
			similarity: 0.95,
		},
		Neighbor{
			word:       "b",
			similarity: 0.9,
		},
		Neighbor{
			word:       "c",
			similarity: 0.85,
		},
		Neighbor{
			word:       "d",
			similarity: 0.8,
		},
	}
	if err := Describe(neighbors); err != nil {
		t.Errorf("Failed to describe neighbors=%v: %s", neighbors, err.Error())
	}
}
