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

	if ns.contextVector != nil {
		t.Error("NegativeSampling: Initializing without building negative vactors")
	}
}

func TestInitialize(t *testing.T) {
	sampleSize := 10
	ns := NewNegativeSampling(sampleSize)

	dimension := 10
	ns.initialize(corpus.TestWord2vecCorpus, dimension)

	expectedVectorSize := corpus.TestWord2vecCorpus.Size() * dimension
	if len(ns.contextVector) != expectedVectorSize {
		t.Errorf("NegativeSampling: Init returns negativeTensor with length=%v: %v",
			expectedVectorSize, len(ns.contextVector))
	}
}
