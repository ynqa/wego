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

package glove

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/word-embedding/corpus"
	"github.com/ynqa/word-embedding/corpus/co"
	"github.com/ynqa/word-embedding/model"
)

// Glove stores the configs for GloVe models.
type Glove struct {
	*model.Config
	*corpus.GloveCorpus

	solver Solver

	pairs  []pairWithFreq
	vector []float64

	xmax  int
	alpha float64

	indexPerThread []int

	progress *pb.ProgressBar
}

// NewGlove creates *Glove.
func NewGlove(f io.ReadCloser, config *model.Config, solver Solver,
	xmax int, alpha float64) *Glove {
	c := corpus.NewGloveCorpus(f, config.ToLower, config.MinCount,
		config.Window, co.IncDist)
	glove := &Glove{
		Config:      config,
		GloveCorpus: c,

		solver: solver,

		alpha: alpha,
		xmax:  xmax,

		indexPerThread: make([]int, config.Thread+1),
	}
	glove.initialize()
	return glove
}

type pairWithFreq struct {
	l1, l2         int
	f, coefficient float64
}

func (g *Glove) buildPairs() {
	coo := g.Cooccurrence()
	g.pairs = make([]pairWithFreq, len(coo))
	shuffle := rand.Perm(len(coo))

	if g.Verbose {
		fmt.Println("Build pairs from corpus:")
		g.progress = pb.New(len(coo)).SetWidth(80)
		g.progress.Start()
	}

	i := 0
	for p, f := range coo {
		coefficient := 1.0
		if f < float64(g.xmax) {
			coefficient = math.Pow(f/float64(g.xmax), g.alpha)
		}

		ul1, ul2 := co.DecodeBigram(p)
		g.pairs[shuffle[i]] = pairWithFreq{
			l1:          int(ul1),
			l2:          int(ul2),
			f:           math.Log(f),
			coefficient: coefficient,
		}
		i++
		if g.Verbose {
			g.progress.Increment()
		}
	}
	if g.Verbose {
		g.progress.Finish()
	}
}

func (g *Glove) initialize() {
	// Build pairs.
	g.buildPairs()

	weightSize := g.Corpus.Size() * (g.Dimension + 1) * 2
	// Initialize word vector.
	g.vector = make([]float64, weightSize)
	for i := 0; i < g.Corpus.Size()*(g.Dimension+1)*2; i++ {
		g.vector[i] = rand.Float64() / float64(g.Dimension)
	}

	// Initialize solver.
	g.solver.initialize(weightSize)
}

// Train trains a corpus.
func (g *Glove) Train() error {
	if len(g.pairs) <= 0 {
		return errors.Errorf("Must initialize model parameters by calling Preprocess")
	}
	if g.Verbose {
		fmt.Printf("Size of Corpus: %v\n", g.Corpus.Size())
		fmt.Printf("Size of Pair: %v\n", len(g.pairs))
	}

	numLines := len(g.pairs)
	g.indexPerThread[0] = 0
	g.indexPerThread[g.Thread] = numLines
	for i := 1; i < g.Thread; i++ {
		g.indexPerThread[i] = g.indexPerThread[i-1] + int(math.Trunc(float64((numLines+i)/g.Thread)))
	}

	sema := make(chan struct{}, g.Thread)
	var wg sync.WaitGroup
	for i := 1; i <= g.Iteration; i++ {
		if g.Verbose {
			fmt.Printf("%d-th:\n", i)
			g.progress = pb.New(len(g.pairs)).SetWidth(80)
			g.progress.Start()
		}
		for j := 0; j < g.Thread; j++ {
			wg.Add(1)
			go g.trainPerThread(g.indexPerThread[j], g.indexPerThread[j+1], &wg, sema)
		}
		wg.Wait()
		g.solver.postOneIter()
		if g.Verbose {
			g.progress.Finish()
		}
	}
	return nil
}

func (g *Glove) trainPerThread(start, end int, wg *sync.WaitGroup, sema chan struct{}) {
	defer wg.Done()
	sema <- struct{}{}
	for i := start; i < end; i++ {
		if g.Verbose {
			g.progress.Increment()
		}

		p := g.pairs[i]
		l1 := p.l1 * (g.Dimension + 1)
		l2 := (p.l2 + g.Corpus.Size()) * (g.Dimension + 1)
		g.solver.trainOne(l1, l2, p.f, p.coefficient, g.vector)
	}
	<-sema
}

// Save saves the word vector to outputFile.
func (g *Glove) Save(outputPath string) error {
	extractDir := func(path string) string {
		e := strings.Split(path, "/")
		return strings.Join(e[:len(e)-1], "/")
	}

	dir := extractDir(outputPath)

	if err := os.MkdirAll("."+string(filepath.Separator)+dir, 0777); err != nil {
		return err
	}

	file, err := os.Create(outputPath)

	if err != nil {
		return err
	}
	w := bufio.NewWriter(file)

	defer func() {
		w.Flush()
		file.Close()
	}()

	var buf bytes.Buffer
	for i := 0; i < g.Corpus.Size(); i++ {
		word, _ := g.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < g.Dimension; j++ {
			l1 := i * (g.Dimension + 1)
			l2 := (i + g.Corpus.Size()) * (g.Dimension + 1)
			fmt.Fprintf(&buf, "%v ", g.vector[l1+j]+g.vector[l2+j])
		}
		fmt.Fprintln(&buf)
	}
	w.WriteString(fmt.Sprintf("%v", buf.String()))
	return nil
}
