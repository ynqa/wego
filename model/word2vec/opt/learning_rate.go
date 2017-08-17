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

package opt

const theta = 1.0e-4

var (
	lr         float64
	initlr     float64
	totalWords int
)

func initLearningRate(v float64) {
	initlr = v
}

func initTotal(v int) {
	totalWords = v
}

func updateLearningRate(currentWords int) {
	lr = initlr * (1.0 - float64(currentWords)/float64(totalWords))
	if lr < initlr*theta {
		lr = initlr * theta
	}
	//fmt.Printf("Learning Rate: %f -> %f in %d/%d\n", initlr, lr, currentWords, totalWords)
}
