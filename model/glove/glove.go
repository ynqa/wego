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
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"

	"github.com/ynqa/word-embedding/model"
	"math/rand"
)

// GloVe stores the configs for GloVe models.
type GloVe struct {
	*model.Config
	*corpus.Corpus

	CofreqMap
	pairs []PairWithFreq

	parameter []float64
	gradsq    []float64

	iteration int
	xmax      int
	alpha     float64

	costPerThread  []float64
	indexPerThread []int

	currentLearningRate float64
}

// NewGloVe creates *GloVe.
func NewGloVe(config *model.Config, iteration int, alpha float64, xmax int) *GloVe {
	// WITHOUT WHITESPACE, UNKNOWN, ROOT
	emptyOpt := func(c *corpus.Corpus) error { return nil }
	c, _ := corpus.Construct(emptyOpt)

	return &GloVe{
		Config: config,
		Corpus: c,

		CofreqMap: make(CofreqMap),

		iteration: iteration,
		alpha:     alpha,
		xmax:      xmax,

		costPerThread:  make([]float64, config.Thread),
		indexPerThread: make([]int, config.Thread+1),

		currentLearningRate: config.InitLearningRate,
	}
}

func (g *GloVe) update(words []string) {
	for i := 0; i < len(words); i++ {
		g.Add(words[i])
		for j := i - g.Config.Window; j <= i+g.Config.Window; j++ {
			if i == j || j < 0 || j >= len(words) {
				continue
			}
			g.CofreqMap.update(words[i], words[j], math.Abs(float64(i-j)))
		}
	}
	return
}

// Preprocess scans the corpus once before Train to count the co-frequency between word-word.
func (g *GloVe) Preprocess(f io.ReadSeeker) (io.ReadCloser, error) {
	defer func() {
		g.parameter = make([]float64, g.Corpus.Size()*(g.Dimension+1)*2)
		g.gradsq = make([]float64, g.Corpus.Size()*(g.Dimension+1)*2)

		for i:=0; i<g.Corpus.Size()*(g.Dimension+1)*2; i++ {
			g.parameter[i] = rand.Float64() / float64(g.Dimension)
			g.gradsq[i] = 1.
		}

		g.pairs = g.CofreqMap.toList(g.Corpus, g.xmax, g.alpha)
	}()

	// For goroutine
	//buffered := 0
	//line := make([]string, 0, 10000)
	//
	//scanner := bufio.NewScanner(f)
	//scanner.Split(bufio.ScanWords)
	//for scanner.Scan() {
	//
	//	if buffered < 10000 {
	//		line = append(line, scanner.Text())
	//		buffered++
	//		continue
	//	}
	//
	//	if g.ToLower {
	//		model.Lower(line)
	//	}
	//
	//	g.update(line)
	//	buffered = 0
	//	line = line[:0]
	//}
	//
	//g.update(line)

	line := make([]string, 0)

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)

	for scanner.Scan() {
		line = append(line, scanner.Text())

		if g.ToLower {
			model.Lower(line)
		}
	}

	g.update(line)

	if err := scanner.Err(); err != nil && err != io.EOF {
		return nil, errors.Wrap(err, "Unable to complete scanning")
	}

	if _, err := f.Seek(0, 0); err != nil {
		return nil, errors.Wrap(err, "Unable to rewind file")
	}
	return f.(io.ReadCloser), nil
}

// Train trains a corpus. It assumes that Preprocess() has already been called.
func (g *GloVe) Train(f io.ReadCloser) error {
	f.Close()
	if len(g.pairs) == 0 {
		return errors.Errorf("Must initialize model parameters by calling Preprocess")
	}

	fmt.Printf("The size of Corpus: %v\n", g.Corpus.Size())
	fmt.Printf("The size of CofreqMap: %v\n", len(g.CofreqMap))
	fmt.Printf("The size of Pairs: %v\n", len(g.pairs))

	numLines := len(g.pairs)
	g.indexPerThread[0] = 0
	for i := 1; i < g.Thread-1; i++ {
		g.indexPerThread[i] = g.indexPerThread[i-1] + numLines/g.Thread
	}
	g.indexPerThread[g.Thread] = numLines - 1

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

		for j, c := range g.costPerThread {
			totalCost += c
			g.costPerThread[j] = 0.
		}

		fmt.Printf("%v-th iteration cost: %v at %v\n", i+1, totalCost/float64(len(g.pairs)), time.Now())
	}
	return nil
}

func (g *GloVe) trainPerThread(tid, start, end int, wg *sync.WaitGroup, sema chan struct{}) {
	defer wg.Done()
	sema <- struct{}{}

	for i := start; i < end; i++ {
		p := g.pairs[i]
		g.train(tid, p.l1, p.l2, p.f, p.coefficient)
	}

	<-sema
}

func (g *GloVe) train(tid, pind, qind int, f, coefficient float64) {

	l1 := pind * (g.Dimension + 1)
	l2 := (qind + g.Corpus.Size()) * (g.Dimension + 1)

	var diff float64
	for i := 0; i < g.Dimension; i++ {
		diff += g.parameter[l1+i] * g.parameter[l2+i]
	}
	diff += g.parameter[l1+g.Dimension] + g.parameter[l2+g.Dimension] - math.Log(f)
	fdiff := diff * coefficient
	g.costPerThread[tid] += 0.5 * fdiff * diff
	fdiff *= g.InitLearningRate

	for i := 0; i < g.Dimension; i++ {
		temp1 := fdiff * g.parameter[l2+i]
		temp2 := fdiff * g.parameter[l1+i]
		g.gradsq[l1+i] += temp1 * temp1
		g.gradsq[l2+i] += temp2 * temp2

		temp1 /= math.Sqrt(g.gradsq[l1+i])
		temp2 /= math.Sqrt(g.gradsq[l2+i])
		g.parameter[l1+i] -= temp1
		g.parameter[l2+i] -= temp2
	}

	g.parameter[l1+g.Dimension] -= fdiff / math.Sqrt(g.gradsq[l1+g.Dimension])
	g.parameter[l2+g.Dimension] -= fdiff / math.Sqrt(g.gradsq[l2+g.Dimension])
	fdiff *= fdiff
	g.gradsq[l1+g.Dimension] += fdiff
	g.gradsq[l2+g.Dimension] += fdiff
}

// Save saves the word vector to outputFile.
func (g *GloVe) Save(outputPath string) error {
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
		for j:=0; j<g.Dimension; j++{
			l1 := i * (g.Dimension + 1)
			l2 := (i + g.Corpus.Size()) * (g.Dimension + 1)
			fmt.Fprintf(&buf, "%v ", g.parameter[l1+j]+g.parameter[l2+j])
		}
		fmt.Fprintln(&buf)
	}
	w.WriteString(fmt.Sprintf("%v", buf.String()))
	return nil
}

func (g *GloVe) toIDs(words []string) []int {
	retVal := make([]int, len(words))
	for i, w := range words {
		retVal[i], _ = g.Id(w)
	}
	return retVal
}
