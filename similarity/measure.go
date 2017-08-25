package similarity

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
