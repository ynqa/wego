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

package cmdutil

import (
	"github.com/spf13/cobra"
)

const (
	defaultInputFile = "example/word_vectors.txt"
	defaultRank      = 10
)

func AddInputFlags(cmd *cobra.Command, input *string) {
	cmd.Flags().StringVarP(input, "input", "i", defaultInputFile, "input file path for trained word vector")
}

func AddRankFlags(cmd *cobra.Command, rank *int) {
	cmd.Flags().IntVarP(rank, "rank", "r", defaultRank, "how many similar words will be displayed")
}
