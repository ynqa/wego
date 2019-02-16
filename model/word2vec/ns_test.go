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
	"testing"

	"github.com/ynqa/wego/corpus"
)

func TestNewNegativeSampling(t *testing.T) {
	sampleSize := 10
	ns := NewNegativeSampling(sampleSize)

	if ns.ContextVector != nil {
		t.Error("NegativeSampling: Initializing without building negative vactors")
	}
}

func TestInitialize(t *testing.T) {
	sampleSize := 10
	ns := NewNegativeSampling(sampleSize)

	dimension := 10
	c := corpus.NewWord2vecCorpus()
	c.Parse(corpus.FakeSeeker, true, 0, 0, false)
	ns.initialize(c, dimension)

	expectedVectorSize := c.Size() * dimension
	if len(ns.ContextVector) != expectedVectorSize {
		t.Errorf("NegativeSampling: Init returns negativeTensor with length=%v: %v",
			expectedVectorSize, len(ns.ContextVector))
	}
}
