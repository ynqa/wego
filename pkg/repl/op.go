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

package repl

type Operator func(float64, float64) float64

func elementWise(v1, v2 []float64, op Operator) []float64 {
	for i := 0; i < len(v1); i++ {
		v1[i] = op(v1[i], v2[i])
	}
	return v1
}

func addOp(x, y float64) float64 {
	return x + y
}

func Add(v1, v2 []float64) []float64 {
	return elementWise(v1, v2, addOp)
}

func subOp(x, y float64) float64 {
	return x - y
}

func Sub(v1, v2 []float64) []float64 {
	return elementWise(v1, v2, subOp)
}
