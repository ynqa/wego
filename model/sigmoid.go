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

package model

import (
	"math"

	"github.com/chewxy/math32"
)

// SigmoidF64 returns f(x) = \frac{1}{1 + e^{-x}}.
// See: http://en.wikipedia.org/wiki/Sigmoid_function.
func SigmoidF64(f float64) float64 {
	exp := math.Exp(f)
	if math.IsInf(exp, 1) {
		return 1.0
	}
	return exp / (1.0 + exp)
}

// SigmoidF32 returns f(x) = \frac{1}{1 + e^{-x}}.
func SigmoidF32(f float32) float32 {
	exp := math32.Exp(f)
	if math32.IsInf(exp, 1) {
		return float32(1.0)
	}
	return exp / (float32(1.0) + exp)

}
