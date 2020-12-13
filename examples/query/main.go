// Copyright © 2020 wego authors
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

package main

import (
	"log"
	"os"

	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/search"
)

func main() {
	input, err := os.Open("word_vector.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer input.Close()
	embs, err := embedding.Load(input)
	if err != nil {
		log.Fatal(err)
	}
	searcher, err := search.New(embs...)
	if err != nil {
		log.Fatal(err)
	}
	neighbors, err := searcher.SearchInternal("given_word", 10)
	if err != nil {
		log.Fatal(err)
	}
	neighbors.Describe()
}
