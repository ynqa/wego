// Copyright © 2020 wego authors
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

package searchutil

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ynqa/wego/pkg/embedding/embutil"
)

func TestCosine(t *testing.T) {
	testCases := []struct {
		name   string
		v1     []float64
		v2     []float64
		expect float64
	}{
		{
			name:   "cosine",
			v1:     []float64{1, 1, 1, 1, 0, 0},
			v2:     []float64{1, 1, 0, 0, 1, 1},
			expect: 0.5,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expect, Cosine(tc.v1, tc.v2, embutil.Norm(tc.v1), embutil.Norm(tc.v2)))
		})
	}
}
