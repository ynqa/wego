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

package query

import (
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"github.com/ynqa/wego/cmd/query/cmdutil"
	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/search"
)

var (
	inputFile string
	rank      int
)

func New() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "query",
		Short:   "Query similar words",
		Example: "  wego query -i example/word_vectors.txt microsoft",
		RunE: func(cmd *cobra.Command, args []string) error {
			return execute(args)
		},
	}
	cmdutil.AddInputFlags(cmd, &inputFile)
	cmdutil.AddRankFlags(cmd, &rank)
	return cmd
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func execute(args []string) error {
	if !fileExists(inputFile) {
		return errors.Errorf("Not such a file %s", inputFile)
	} else if len(args) != 1 {
		return errors.Errorf("Input a single word %v", args)
	}
	input, err := os.Open(inputFile)
	if err != nil {
		return err
	}
	defer input.Close()
	embs, err := embedding.Load(input)
	if err != nil {
		return err
	}
	searcher, err := search.New(embs...)
	if err != nil {
		return err
	}
	neighbors, err := searcher.SearchInternal(args[0], rank)
	if err != nil {
		return err
	}
	neighbors.Describe()
	return nil
}
