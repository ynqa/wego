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

package repl

import (
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"github.com/ynqa/wego/cmd/search/cmdutil"
	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/search"
	"github.com/ynqa/wego/pkg/search/repl"
)

var (
	inputFile string
	rank      int
)

func New() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "search-repl",
		Short: "Search similar words (REPL mode)",
		Long:  "Search similar words (REPL mode)",
		Example: "  wego search-repl -i example/word_vectors.txt\n" +
			"  >> apple + banana\n" +
			"  ...",
		RunE: func(cmd *cobra.Command, args []string) error {
			return execute()
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

func execute() error {
	if !fileExists(inputFile) {
		return errors.Errorf("Not such a file %s", inputFile)
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
	repl, err := repl.New(searcher, rank)
	if err != nil {
		return err
	}
	return repl.Run()
}
