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
)

func TestNewNegativeSampling(t *testing.T) {
	ns := NewNegativeSampling(10)

	if ns.contextVector != nil {
		t.Error("NegativeSampling: Initializing without building negative vactors")
	}
}

func TestNSInit(t *testing.T) {
	ns := NewNegativeSampling(10)
	ns.init(newTestCorpus(), 10)

	if len(ns.contextVector) != 3*10 {
		t.Error("NegativeSampling: Init returns negativeTensor with length=3*10")
	}
}
