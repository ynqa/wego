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
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/model"
	"github.com/ynqa/wego/pkg/model/glove"
	"github.com/ynqa/wego/pkg/model/lexvec"
	"github.com/ynqa/wego/pkg/model/modelutil/vector"
	"github.com/ynqa/wego/pkg/model/word2vec"
	"github.com/ynqa/wego/pkg/search"
)

const (
	text8 = "./test/testdata/text8"
	query = "microsoft"
)

func unwrap(mod model.Model, err error) model.Model {
	return mod
}

func main() {
	testcases := []struct {
		title string
		mod   model.Model
	}{
		{
			title: "word2vec (model=skip-gram, optimizer=negative sampling)",
			mod: unwrap(word2vec.New(
				word2vec.BatchSize(10000),
				word2vec.Dim(50),
				word2vec.Goroutines(20),
				word2vec.Iter(1),
				word2vec.MinCount(10),
				word2vec.Model(word2vec.SkipGram),
				word2vec.Optimizer(word2vec.NegativeSampling),
				word2vec.Verbose(),
				word2vec.Window(5),
			)),
		},
		{
			title: "word2vec (model=skip-gram, optimizer=hierarchical softmax)",
			mod: unwrap(word2vec.New(
				word2vec.BatchSize(10000),
				word2vec.Dim(50),
				word2vec.Goroutines(20),
				word2vec.Iter(1),
				word2vec.MinCount(10),
				word2vec.Model(word2vec.SkipGram),
				word2vec.Optimizer(word2vec.HierarchicalSoftmax),
				word2vec.Verbose(),
				word2vec.Window(5),
			)),
		},
		{
			title: "word2vec (model=cbow, optimizer=negative sampling)",
			mod: unwrap(word2vec.New(
				word2vec.BatchSize(10000),
				word2vec.Dim(50),
				word2vec.Goroutines(20),
				word2vec.Iter(1),
				word2vec.MinCount(10),
				word2vec.Model(word2vec.Cbow),
				word2vec.Optimizer(word2vec.NegativeSampling),
				word2vec.Verbose(),
				word2vec.Window(5),
			)),
		},
		{
			title: "word2vec (model=cbow, optimizer=hierarchical softmax)",
			mod: unwrap(word2vec.New(
				word2vec.BatchSize(10000),
				word2vec.Dim(50),
				word2vec.Goroutines(20),
				word2vec.Iter(1),
				word2vec.MinCount(10),
				word2vec.Model(word2vec.Cbow),
				word2vec.Optimizer(word2vec.HierarchicalSoftmax),
				word2vec.Verbose(),
				word2vec.Window(5),
			)),
		},
		{
			title: "glove (solver=sgd)",
			mod: unwrap(glove.New(
				glove.BatchSize(10000),
				glove.Dim(50),
				glove.Goroutines(20),
				glove.Initlr(0.01),
				glove.Iter(3),
				glove.MinCount(10),
				glove.Solver(glove.Stochastic),
				glove.Verbose(),
				glove.Window(10),
			)),
		},
		{
			title: "glove (solver=adagrad)",
			mod: unwrap(glove.New(
				glove.BatchSize(10000),
				glove.Dim(50),
				glove.Goroutines(20),
				glove.Initlr(0.01),
				glove.Iter(3),
				glove.MinCount(10),
				glove.Solver(glove.AdaGrad),
				glove.Verbose(),
				glove.Window(10),
			)),
		},
		{
			title: "lexvec",
			mod: unwrap(lexvec.New(
				lexvec.BatchSize(10000),
				lexvec.Dim(50),
				lexvec.Goroutines(20),
				lexvec.Iter(1),
				lexvec.MinCount(10),
				lexvec.Relation(lexvec.PPMI),
				lexvec.Verbose(),
				lexvec.Window(10),
			)),
		},
	}
	for _, tt := range testcases {
		fmt.Printf("test in %s\n", tt.title)
		if err := e2e(tt.mod); err != nil {
			log.Fatal(err)
		}
	}
}

func e2e(mod model.Model) error {
	input, err := os.Open(text8)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := ioutil.TempFile("", "wego")
	if err != nil {
		log.Fatal(err)
	}
	if err := mod.Train(input); err != nil {
		return err
	}
	if err := mod.Save(output, vector.Agg); err != nil {
		return err
	}

	output.Seek(0, 0)

	embs, err := embedding.Load(output)
	if err != nil {
		return err
	}
	searcher, err := search.New(embs...)
	if err != nil {
		return err
	}
	neighbors, err := searcher.SearchInternal(query, 10)
	if err != nil {
		return err
	}
	neighbors.Describe()

	return nil
}
