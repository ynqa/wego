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

package corpus

import (
	"github.com/spf13/cobra"
)

const (
	defaultDocInMemory = false
	defaultToLower     = false
)

type Options struct {
	DocInMemory bool
	ToLower     bool
}

func DefaultOptions() Options {
	return Options{
		DocInMemory: defaultDocInMemory,
		ToLower:     defaultToLower,
	}
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().BoolVar(&opts.DocInMemory, "in-memory", defaultDocInMemory, "whether to store the doc in memory")
	cmd.Flags().BoolVar(&opts.ToLower, "lower", defaultToLower, "whether the words on corpus convert to lowercase or not")
}
