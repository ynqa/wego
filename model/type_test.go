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

package model

import (
	"testing"
)

func TestRandomTensor(t *testing.T) {
	d, err := NewType("float64")

	if err != nil {
		t.Error(err.Error())
	}

	tns := d.RandomTensor(20, 10)

	if tns.Shape()[0] != 20 {
		t.Errorf("RandomTensor(20, 10) returns a tensor with 20 rows, but %v", tns.Shape()[0])
	} else if tns.Shape()[1] != 10 {
		t.Errorf("RandomTensor(20, 10) returns a tensor with 10 columns, but %v", tns.Shape()[1])
	}
}
