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

package word2vec

import (
	"bytes"
	"io"
	"io/ioutil"
	"strings"
	"testing"

	"github.com/chewxy/lingo/corpus"

	"github.com/ynqa/word-embedding/config"
	"github.com/ynqa/word-embedding/model"
)

// MockOperator satisfies the interface of Optimizer.
type MockOptimizer struct{}

func (m *MockOptimizer) Init(c *corpus.Corpus, dimension int) error { return nil }
func (m *MockOptimizer) Update(targetID int, contextVector, poolVector []float64, learningRate float64) {
}

// MockNopSeeker stores io.ReadCloser with Seek func that has nothing.
type MockNopSeeker struct{ io.ReadCloser }

func (n MockNopSeeker) Seek(offset int64, whence int) (int64, error) { return 0, nil }

var (
	text = "A B B C C C C"
	conf = model.NewConfig(
		config.DefaultDimension,
		config.DefaultWindow,
		config.DefaultInitLearningRate,
		config.DefaultThread,
		config.DefaultToLower,
		config.DefaultVerbose,
	)
	mockOpt       Optimizer = new(MockOptimizer)
	mockNopSeeker           = MockNopSeeker{ReadCloser: ioutil.NopCloser(bytes.NewReader([]byte(text)))}
)

func newTestCorpus() *corpus.Corpus {
	var emptyOpt corpus.ConsOpt = func(c *corpus.Corpus) error { return nil }

	c, _ := corpus.Construct(emptyOpt)
	for _, word := range strings.Fields(text) {
		c.Add(word)
	}
	return c
}

// newTestState returns *State has already called Preprocess.
func newTestState() *State {
	s := NewState(
		conf,
		mockOpt,
		config.DefaultSubsampleThreshold,
		config.DefaultTheta,
		config.DefaultBatchSize,
	)
	s.Preprocess(mockNopSeeker)
	return s
}

func TestPreprocess(t *testing.T) {
	testState := NewState(
		conf,
		mockOpt,
		config.DefaultSubsampleThreshold,
		config.DefaultTheta,
		config.DefaultBatchSize,
	)

	if _, err := testState.Preprocess(mockNopSeeker); err != nil {
		t.Error("Word2Vec: Preprocess returns error")
	}
}
