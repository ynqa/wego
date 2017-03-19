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
	"github.com/ynqa/word-embedding/models"
	"github.com/ynqa/word-embedding/utils/fileio"
)

type Word2Vec struct {
	models.Common
	Model Model
	Opt   Optimizer
}

func (w Word2Vec) PreTrain() error {
	return w.Opt.PreTrain()
}

func (w Word2Vec) Run() error {
	if err := fileio.Load(w.Common.InputFile, w.iterator); err != nil {
		return err
	}
	return nil
}

func (w Word2Vec) iterator(words []string) {
	for index := range words {
		w.Model.Train(words, index, w.Opt.Update)
	}
}

func (w Word2Vec) Save() error {
	return GlobalWordMap.Save(w.Common.OutputFile)
}
