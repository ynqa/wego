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
	"io"
	"io/ioutil"
	"testing"

	"github.com/ynqa/word-embedding/config"
	"github.com/ynqa/word-embedding/model"
)

// MockSolver satisfies the interface of Solver.
type MockSolver struct{}

func (m *MockSolver) init(weightSie int)                                                    {}
func (m *MockSolver) trainOne(l1, l2 int, f, coefficient float64, weight []float64) float64 { return 0. }
func (m *MockSolver) callback()                                                             {}

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
	mockSolver    = new(MockSolver)
	mockNopSeeker = MockNopSeeker{ReadCloser: ioutil.NopCloser(bytes.NewReader([]byte(text)))}
)

func TestGloVePreprocess(t *testing.T) {
	testGloVe := NewGloVe(
		conf,
		mockSolver,
		config.DefaultIteration,
		config.DefaultXmax,
		config.DefaultAlpha,
		config.DefaultMinCount,
		config.DefaultBatchSize,
	)
	if _, err := testGloVe.Preprocess(mockNopSeeker); err != nil {
		t.Error("Word2Vec: Preprocess returns error")
	}
}
