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
	"github.com/ynqa/word-embedding/model/word2vec/huffman"
	"github.com/ynqa/word-embedding/utils"
)

var (
	// GlobalFreqMap stores the words' frequency in corpus.
	GlobalFreqMap utils.FreqMap
	// GlobalWordMap stores the word map.
	GlobalWordMap WordMap
	// GlobalNodeMap stores the node map.
	GlobalNodeMap huffman.NodeMap
)

var (
	globalVocabulary int
	globalWords      int
)

// GetVocabulary returns the vocabulary size.
func GetVocabulary() int {
	if globalVocabulary == 0 {
		globalVocabulary = GlobalFreqMap.Vocabulary()
	}
	return globalVocabulary
}

// GetWords returns the number of words.
func GetWords() int {
	if globalWords == 0 {
		globalWords = GlobalFreqMap.Words()
	}
	return globalWords
}
