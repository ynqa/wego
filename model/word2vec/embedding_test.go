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

func TestNewEmbedding(t *testing.T) {
	tns := NewEmbedding(typ, 20, 10)

	if len(tns.m) != 20 {
		t.Errorf("NewEmbedding(20, 10) returns a tensor with 20 vocabularies, but %v", len(tns.m))
	} else if tns.m[0].Shape()[0] != 10 {
		t.Errorf("NewEmbedding(20, 10) returns a tensor with 20 dimensions, but %v", tns.m[0].Shape()[0])
	}
}
