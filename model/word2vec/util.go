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

var next uint64 = 1

// Linear congruential generator like rand.Intn(window)
func nextRandom(window int) int {
	next = next*uint64(25214903917) + 11
	return int(next % uint64(window))
}

func updateLearningRate(initlr, theta float64, currentWords, totalWords int) float64 {
	lr := initlr * (1.0 - float64(currentWords)/float64(totalWords))
	if lr < initlr*theta {
		lr = initlr * theta
	}
	return lr
}
