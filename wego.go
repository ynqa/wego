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
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"github.com/ynqa/wego/cmd/model/glove"
	"github.com/ynqa/wego/cmd/model/lexvec"
	"github.com/ynqa/wego/cmd/model/word2vec"
	"github.com/ynqa/wego/cmd/query"
	"github.com/ynqa/wego/cmd/query/console"
)

func main() {
	word2vec := word2vec.New()
	glove := glove.New()
	lexvec := lexvec.New()
	query := query.New()
	console := console.New()

	cmd := &cobra.Command{
		Use:   "wego",
		Short: "tools for embedding words into vector space",
		RunE: func(cmd *cobra.Command, args []string) error {
			return errors.Errorf("Set sub-command. One of %s|%s|%s|%s|%s",
				word2vec.Name(),
				glove.Name(),
				lexvec.Name(),
				query.Name(),
				console.Name(),
			)
		},
	}
	cmd.AddCommand(word2vec)
	cmd.AddCommand(glove)
	cmd.AddCommand(lexvec)
	cmd.AddCommand(query)
	cmd.AddCommand(console)

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
