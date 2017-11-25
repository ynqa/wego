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
	"time"

	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"

	"github.com/ynqa/word-embedding/model"
)

// Glove stores the configs for GloVe models.
type Glove struct {
	*model.Config
	*corpus.Corpus

	CofreqMap
	pairs []PairWithFreq

	solver Solver

	weight []float64

	iteration int
	xmax      int
	alpha     float64

	minCount  int
	batchSize int

	costPerThread  []float64
	indexPerThread []int
}

// NewGlove creates *Glove.
func NewGlove(config *model.Config, solver Solver,
	iteration int, xmax int, alpha float64, minCount, batchSize int) *Glove {

	// WITHOUT WHITESPACE, UNKNOWN, ROOT
	emptyOpt := func(c *corpus.Corpus) error { return nil }
	c, _ := corpus.Construct(emptyOpt)

	return &Glove{
		Config: config,
		Corpus: c,

		CofreqMap: make(CofreqMap),

		solver: solver,

		iteration: iteration,
		alpha:     alpha,
		xmax:      xmax,

		minCount:  minCount,
		batchSize: batchSize,

		costPerThread:  make([]float64, config.Thread),
		indexPerThread: make([]int, config.Thread+1),
	}
}

func (g *Glove) update(wordIDs []int) {
	for i := 0; i < len(wordIDs); i++ {
		for j := i - g.Config.Window; j <= i+g.Config.Window; j++ {
			if i == j || j < 0 || j >= len(wordIDs) {
				continue
			}
			g.CofreqMap.update(wordIDs[i], wordIDs[j], math.Abs(float64(i-j)))
		}
	}
}

// Preprocess scans the corpus once before Train to count the co-frequency between word-word.
func (g *Glove) Preprocess(f io.ReadSeeker) (io.ReadCloser, error) {
	defer func() {
		weightSize := g.Corpus.Size() * (g.Dimension + 1) * 2

		g.solver.init(weightSize)

		g.weight = make([]float64, weightSize)
		for i := 0; i < g.Corpus.Size()*(g.Dimension+1)*2; i++ {
			g.weight[i] = rand.Float64() / float64(g.Dimension)
		}

		g.pairs = g.CofreqMap.toList(g.Corpus, g.xmax, g.alpha, g.minCount)
	}()

	buffered := 0
	wordIDs := make([]int, g.batchSize)

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)
	for scanner.Scan() {
		word := scanner.Text()
		if g.ToLower {
			word = strings.ToLower(word)
		}
		g.Add(word)
		wordID, _ := g.Id(word)
		wordIDs[buffered] = wordID

		if buffered < g.batchSize-1 {
			buffered++
			continue
		}

		g.update(wordIDs)
		buffered = 0
		wordIDs = make([]int, g.batchSize)
	}

	if buffered > 0 {
		g.update(wordIDs[:buffered])
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return nil, errors.Wrap(err, "Unable to complete scanning")
	}

	if _, err := f.Seek(0, 0); err != nil {
		return nil, errors.Wrap(err, "Unable to rewind file")
	}
	return f.(io.ReadCloser), nil
}

// Train trains a corpus. It assumes that Preprocess() has already been called.
func (g *Glove) Train(f io.ReadCloser) error {
	f.Close()
	if len(g.pairs) == 0 {
		return errors.Errorf("Must initialize model parameters by calling Preprocess")
	}

	if g.Verbose {
		fmt.Printf("The size of Corpus: %v\n", g.Corpus.Size())
		fmt.Printf("The size of Pairs: %v\n", len(g.pairs))
	}

	numLines := len(g.pairs)
	g.indexPerThread[0] = 0
	g.indexPerThread[g.Thread] = numLines
	for i := 1; i < g.Thread; i++ {
		g.indexPerThread[i] = g.indexPerThread[i-1] + int(math.Trunc(float64((numLines+i)/g.Thread)))
	}

	g.costPerThread = make([]float64, g.Thread)

	sema := make(chan struct{}, g.Thread)
	var wg sync.WaitGroup

	for i := 0; i < g.iteration; i++ {
		totalCost := 0.
		for j := 0; j < g.Thread; j++ {
			wg.Add(1)
			go g.trainPerThread(j, g.indexPerThread[j], g.indexPerThread[j+1], &wg, sema)
		}
		wg.Wait()

		g.solver.callback()

		if g.Verbose {
			for j, c := range g.costPerThread {
				totalCost += c
				g.costPerThread[j] = 0.
			}
			fmt.Printf("%v-th iteration cost: %v at %v\n", i+1, totalCost/float64(len(g.pairs)), time.Now())
		}
	}
	return nil
}

func (g *Glove) trainPerThread(threadID, start, end int, wg *sync.WaitGroup, sema chan struct{}) {
	defer wg.Done()
	sema <- struct{}{}

	for i := start; i < end; i++ {
		p := g.pairs[i]
		l1 := p.l1 * (g.Dimension + 1)
		l2 := (p.l2 + g.Corpus.Size()) * (g.Dimension + 1)
		g.costPerThread[threadID] += g.solver.trainOne(l1, l2, p.f, p.coefficient, g.weight)
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
			fmt.Fprintf(&buf, "%v ", g.weight[l1+j]+g.weight[l2+j])
		}
		fmt.Fprintln(&buf)
	}
	w.WriteString(fmt.Sprintf("%v", buf.String()))
	return nil
}
