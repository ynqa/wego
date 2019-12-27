package matrix

type Matrix struct {
	array []float64
	row   int
	col   int
}

func New(row, col int, fn func([]float64)) *Matrix {
	mat := &Matrix{
		array: make([]float64, row*col),
		row:   row,
		col:   col,
	}
	for i := 0; i < row; i++ {
		fn(mat.Slice(i))
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
