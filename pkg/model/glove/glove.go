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
	"bytes"
	"fmt"
	"io"
	"math/rand"
	"sync"

	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/ynqa/wego/pkg/corpus"
	"github.com/ynqa/wego/pkg/model"
)

type GloveOption struct {
	Solver Solver
	Xmax   int
	Alpha  float64
}

// Glove stores the configs for Glove models.
type Glove struct {
	*model.Option
	*GloveOption
	*corpus.CountModelCorpus

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
func NewGlove(option *model.Option, gloveOption *GloveOption) *Glove {
	return &Glove{
		Option:      option,
		GloveOption: gloveOption,
	}
}

func (g *Glove) initialize() (err error) {
	// Build pairs based on co-occurrence.
	g.pairs, err = g.CountModelCorpus.PairsIntoGlove(g.Window, g.Xmax, g.Alpha, g.Verbose)
	if err != nil {
		return errors.Wrapf(err, "Failed to initialize for GloVe")
	}

	// Initialize word vector.
	vectorSize := g.CountModelCorpus.Size() * (g.Dimension + 1) * 2
	g.vector = make([]float64, vectorSize)
	for i := 0; i < vectorSize; i++ {
		g.vector[i] = rand.Float64() / float64(g.Dimension)
	}

	// Initialize solver.
	switch solver := g.Solver.(type) {
	case *AdaGrad:
		solver.initialize(vectorSize)
	}
	return nil
}

// Train trains words' vector on corpus.
func (g *Glove) Train(f io.Reader) error {
	c := corpus.NewCountModelCorpus()
	if err := c.Parse(f, g.ToLower, g.MinCount, g.BatchSize, g.Verbose); err != nil {
		return errors.Wrap(err, "Unable to generate *Glove")
	}
	g.CountModelCorpus = c
	if err := g.initialize(); err != nil {
		return errors.Wrap(err, "Failed to initialize")
	}
	return g.train()
}

func (g *Glove) train() error {
	pairSize := len(g.pairs)
	if pairSize <= 0 {
		return errors.Errorf("No pairs for training")
	}

	g.indexPerThread = model.IndexPerThread(g.ThreadSize, pairSize)

	semaphore := make(chan struct{}, g.ThreadSize)
	waitGroup := &sync.WaitGroup{}

	for i := 1; i <= g.Iteration; i++ {
		if g.Verbose {
			fmt.Printf("Train %d-th:\n", i)
			g.progress = pb.New(pairSize).SetWidth(80)
			g.progress.Start()
		}

		for j := 0; j < g.ThreadSize; j++ {
			waitGroup.Add(1)
			go g.trainPerThread(g.indexPerThread[j], g.indexPerThread[j+1],
				semaphore, waitGroup)
		}

		switch solver := g.Solver.(type) {
		case *Sgd:
			solver.postOneIter()
		}

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
		if g.Verbose {
			g.progress.Increment()
		}
		pair := g.pairs[i]
		l1 := pair.L1 * (g.Dimension + 1)
		l2 := (pair.L2 + g.CountModelCorpus.Size()) * (g.Dimension + 1)
		g.Solver.trainOne(l1, l2, pair.F, pair.Coefficient, g.vector)
		ll1 := (pair.L1 + g.CountModelCorpus.Size()) * (g.Dimension + 1)
		ll2 := pair.L2 * (g.Dimension + 1)
		g.Solver.trainOne(ll1, ll2, pair.F, pair.Coefficient, g.vector)
	}
}

// Save saves the word vector to output writer.
func (g *Glove) Save(output io.Writer) error {
	if output == nil {
		return errors.New("Invalid output writer: must not be nil")
	}

	wordSize := g.CountModelCorpus.Size()
	if g.Verbose {
		fmt.Println("Save:")
		g.progress = pb.New(wordSize).SetWidth(80)
		defer g.progress.Finish()
		g.progress.Start()
	}

	var buf bytes.Buffer
	for i := 0; i < wordSize; i++ {
		word, _ := g.CountModelCorpus.Word(i)
		fmt.Fprintf(&buf, "%v ", word)
		for j := 0; j < g.Dimension; j++ {
			l1 := i*(g.Dimension+1) + j
			var v float64
			switch g.SaveVectorType {
			case model.NORMAL:
				v = g.vector[l1]
			case model.ADD:
				l2 := (i+wordSize)*(g.Dimension+1) + j
				v = g.vector[l1] + g.vector[l2]
			default:
				return errors.Errorf("Invalid save vector type=%s", g.SaveVectorType)
			}

			fmt.Fprintf(&buf, "%v ", v)
		}
		fmt.Fprintln(&buf)
		if g.Verbose {
			g.progress.Increment()
		}
	}

	output.Write(buf.Bytes())
	return nil
}
