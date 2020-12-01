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
	"fmt"

	"github.com/spf13/cobra"

	"github.com/ynqa/wego/pkg/model/modelutil/vector"
)

const (
	defaultInputFile  = "example/input.txt"
	defaultOutputFile = "example/word_vectors.txt"
	defaultProf       = false
	defaultVectorType = vector.Single
)

func AddInputFlags(cmd *cobra.Command, input *string) {
	cmd.Flags().StringVarP(input, "input", "i", defaultInputFile, "input file path for corpus")
}

func AddOutputFlags(cmd *cobra.Command, output *string) {
	cmd.Flags().StringVarP(output, "output", "o", defaultOutputFile, "output file path to save word vectors")
}

func AddProfFlags(cmd *cobra.Command, prof *bool) {
	cmd.Flags().BoolVar(prof, "prof", defaultProf, "profiling mode to check the performances")
}

func AddVectorTypeFlags(cmd *cobra.Command, typ *vector.Type) {
	cmd.Flags().StringVar(typ, "vec-type", defaultVectorType, fmt.Sprintf("word vector type. One of: %s|%s", vector.Single, vector.Agg))
}
