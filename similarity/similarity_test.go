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

package similarity

import (
	"bytes"
	"io/ioutil"
	"testing"
)

var mockVector = `apple 1:1 2:1 3:1 4:1 5:1
	banana 1:1 2:1 3:1 4:1 5:1
	chocolate 1:0 2:0 3:0 4:0 5:0
	dragon 1:-1 2:-1 3:-1 4:-1 5:-1`

func TestEstimate(t *testing.T) {
	estimator := NewEstimator("apple", 3)

	f := ioutil.NopCloser(bytes.NewReader([]byte(mockVector)))
	err := estimator.Estimate(f)

	if err != nil {
		t.Errorf(err.Error())
	}

	if len(estimator.dense) != 4 {
		t.Errorf("Expected estimator.tensor len=4: %d", len(estimator.dense))
	}
}
