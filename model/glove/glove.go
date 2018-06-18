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

	"github.com/ynqa/wego/corpus"
	"github.com/ynqa/wego/corpus/co"
	"github.com/ynqa/wego/model"
)

// Glove stores the configs for Glove models.
type Glove struct {
	*model.Config
	*corpus.GloveCorpus

	solver Solver

	// given parameters.
	xmax  int
	alpha float64

	// word pair with co-occurrence.
	pairs []pair

	// words' vector.
	vector []float64

	// manage data range per thread.
	indexPerThread []int

	// progress bar.
	progress *pb.ProgressBar
}

// NewGlove creates *Glove.
func NewGlove(f io.ReadCloser, config *model.Config, solver Solver,
	xmax int, alpha float64) (*Glove, error) {
	cps, err := corpus.NewGloveCorpus(f, config.ToLower, config.MinCount, config.Window)
	if err != nil {
		return nil, errors.Wrap(err, "Unable to generate *Glove")
	}
	glove := &Glove{
		Config:      config,
		GloveCorpus: cps,

		solver: solver,

		xmax:  xmax,
		alpha: alpha,
	}
	glove.initialize()
	return glove, nil
}

func (g *Glove) initialize() {
	// Build pairs based on co-occurrence.
	g.buildPairs()

	// Initialize word vector.
	vectorSize := g.GloveCorpus.Size() * (g.Config.Dimension + 1) * 2
	g.vector = make([]float64, vectorSize)
	for i := 0; i < vectorSize; i++ {
		g.vector[i] = rand.Float64() / float64(g.Config.Dimension)
	}

	// Initialize solver.
	g.solver.initialize(vectorSize)
}

type pair struct {
	l1, l2         int
	f, coefficient float64
}

func (g *Glove) buildPairs() {
	coo := g.Cooccurrence()
	pairSize := len(coo)
	g.pairs = make([]pair, pairSize)
	shuffle := rand.Perm(pairSize)

	if g.Verbose {
		fmt.Println("Build co-occurrence map from corpus:")
		g.progress = pb.New(pairSize).SetWidth(80)
		g.progress.Start()
	}

	i := 0
	for p, f := range coo {
		coefficient := 1.0
		if f < float64(g.xmax) {
			coefficient = math.Pow(f/float64(g.xmax), g.alpha)
		}

		ul1, ul2 := co.DecodeBigram(p)
		g.pairs[shuffle[i]] = pair{
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

// Train trains words' vector on corpus.
func (g *Glove) Train() error {
	pairSize := len(g.pairs)
	if pairSize <= 0 {
		return errors.Errorf("No pairs for training")
	}
	if g.Config.Verbose {
		fmt.Printf("Size of Corpus: %v\n", g.GloveCorpus.Size())
		fmt.Printf("Size of Pair: %v\n", len(g.pairs))
	}

	g.indexPerThread = model.IndexPerThread(g.Config.ThreadSize, pairSize)

	semaphore := make(chan struct{}, g.Config.ThreadSize)
	waitGroup := &sync.WaitGroup{}

	for i := 1; i <= g.Iteration; i++ {
		if g.Verbose {
			fmt.Printf("%d-th:\n", i)
			g.progress = pb.New(pairSize).SetWidth(80)
			g.progress.Start()
		}

		for j := 0; j < g.Config.ThreadSize; j++ {
			waitGroup.Add(1)
			go g.trainPerThread(g.indexPerThread[j], g.indexPerThread[j+1],
				semaphore, waitGroup)
		}
		g.solver.postOneIter()

		waitGroup.Wait()
		if g.Verbose {
			g.progress.Finish()
		}
	}
	return nil
}

func (g *Glove) trainPerThread(beginIdx, endIdx int,
	semaphore chan struct{}, waitGroup *sync.WaitGroup) {

	defer func() {
		<-semaphore
		waitGroup.Done()
	}()

	semaphore <- struct{}{}
	for i := beginIdx; i < endIdx; i++ {
		if g.Config.Verbose {
			g.progress.Increment()
		}
		pair := g.pairs[i]
		l1 := pair.l1 * (g.Config.Dimension + 1)
		l2 := (pair.l2 + g.Corpus.Size()) * (g.Config.Dimension + 1)
		g.solver.trainOne(l1, l2, pair.f, pair.coefficient, g.vector)
	}
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
	for i := 0; i < g.GloveCorpus.Size(); i++ {
		word, _ := g.GloveCorpus.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < g.Config.Dimension; j++ {
			l1 := i * (g.Config.Dimension + 1)
			l2 := (i + g.GloveCorpus.Size()) * (g.Config.Dimension + 1)
			fmt.Fprintf(&buf, "%v ", g.vector[l1+j]+g.vector[l2+j])
		}
		fmt.Fprintln(&buf)
	}
	w.WriteString(fmt.Sprintf("%v", buf.String()))
	return nil
}
