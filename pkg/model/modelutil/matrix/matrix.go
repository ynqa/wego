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

package matrix

type Matrix struct {
	array []float64
	row   int
	col   int
}

func New(row, col int, fn func(int, []float64)) *Matrix {
	mat := &Matrix{
		array: make([]float64, row*col),
		row:   row,
		col:   col,
	}
	for i := 0; i < row; i++ {
		fn(i, mat.Slice(i))
	}
	return mat
}

func (m *Matrix) startIndex(id int) int {
	return id * m.col
}

func (m *Matrix) Row() int {
	return m.row
}

func (m *Matrix) Col() int {
	return m.col
}

func (m *Matrix) Slice(id int) []float64 {
	start := m.startIndex(id)
	return m.array[start : start+m.col]
}
