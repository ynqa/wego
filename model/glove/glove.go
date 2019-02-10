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
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/wego/corpus"
	"github.com/ynqa/wego/model"
)

// Glove stores the configs for Glove models.
type Glove struct {
	*model.Config
	*corpus.CountModelCorpus

	solver Solver

	// given parameters.
	xmax  int
	alpha float64

	// word pairs.
	pairs []corpus.Pair

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
	cps, err := corpus.NewCountModelCorpus(f, config.ToLower, config.MinCount)
	if err != nil {
		return nil, errors.Wrap(err, "Unable to generate *Glove")
	}
	glove := &Glove{
		Config:           config,
		CountModelCorpus: cps,

		solver: solver,

		xmax:  xmax,
		alpha: alpha,
	}
	if err := glove.initialize(); err != nil {
		return nil, errors.Wrap(err, "Unable to generate *Glove")
	}
	return glove, nil
}

func (g *Glove) initialize() (err error) {
	// Build pairs based on co-occurrence.
	g.pairs, err = g.CountModelCorpus.PairsIntoGlove(g.Window, g.xmax, g.alpha, g.Verbose)
	if err != nil {
		return errors.Wrapf(err, "Failed to initialize for GloVe")
	}

	// Initialize word vector.
	vectorSize := g.CountModelCorpus.Size() * (g.Config.Dimension + 1) * 2
	g.vector = make([]float64, vectorSize)
	for i := 0; i < vectorSize; i++ {
		g.vector[i] = rand.Float64() / float64(g.Config.Dimension)
	}

	// Initialize solver.
	g.solver.initialize(vectorSize)
	return nil
}

// Train trains words' vector on corpus.
func (g *Glove) Train() error {
	pairSize := len(g.pairs)
	if pairSize <= 0 {
		return errors.Errorf("No pairs for training")
	}

	g.indexPerThread = model.IndexPerThread(g.Config.ThreadSize, pairSize)

	semaphore := make(chan struct{}, g.Config.ThreadSize)
	waitGroup := &sync.WaitGroup{}

	for i := 1; i <= g.Iteration; i++ {
		if g.Verbose {
			fmt.Printf("Train %d-th:\n", i)
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
		if g.Config.Verbose {
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
		l1 := pair.L1 * (g.Config.Dimension + 1)
		l2 := (pair.L2 + g.CountModelCorpus.Size()) * (g.Config.Dimension + 1)
		g.solver.trainOne(l1, l2, pair.F, pair.Coefficient, g.vector)
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

	wordSize := g.CountModelCorpus.Size()
	if g.Config.Verbose {
		fmt.Println("Save:")
		g.progress = pb.New(wordSize).SetWidth(80)
		defer g.progress.Finish()
		g.progress.Start()
	}

	var buf bytes.Buffer
	for i := 0; i < wordSize; i++ {
		word, _ := g.CountModelCorpus.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < g.Config.Dimension; j++ {
			l1 := i*(g.Config.Dimension+1) + j
			var v float64
			switch g.SaveVectorType {
			case model.NORMAL:
				v = g.vector[l1]
			case model.ADD:
				l2 := (i+wordSize)*(g.Config.Dimension+1) + j
				v = g.vector[l1] + g.vector[l2]
			default:
				return errors.Errorf("Invalid save vector type=%s", g.SaveVectorType)
			}

			fmt.Fprintf(&buf, "%v ", v)
		}
		fmt.Fprintln(&buf)
		if g.Config.Verbose {
			g.progress.Increment()
		}
	}
	w.WriteString(fmt.Sprintf("%v", buf.String()))
	return nil
}
