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

package embutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNorm(t *testing.T) {
	testCases := []struct {
		name   string
		vec    []float64
		expect float64
	}{
		{
			name:   "norm",
			vec:    []float64{1, 1, 1, 1, 0, 0},
			expect: 2.,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expect, Norm(tc.vec))
		})
	}
}
