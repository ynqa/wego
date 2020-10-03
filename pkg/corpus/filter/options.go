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

package filter

import (
	"github.com/spf13/cobra"

	"github.com/ynqa/wego/pkg/corpus/dictionary"
)

var (
	defaultMaxCount = -1
	defaultMinCount = 5
)

type Options struct {
	MaxCount int
	MinCount int
}

func DefaultOption() *Options {
	return &Options{
		MaxCount: defaultMaxCount,
		MinCount: defaultMinCount,
	}
}

func LoadForCmd(cmd *cobra.Command, opts *Options) {
	cmd.Flags().IntVar(&opts.MaxCount, "max-count", defaultMaxCount, "upper limit to filter words")
	cmd.Flags().IntVar(&opts.MinCount, "min-count", defaultMinCount, "lower limit to filter words")
}

type FilterFn func(id int, dic *dictionary.Dictionary) bool

func MaxCount(v int) FilterFn {
	return FilterFn(func(id int, dic *dictionary.Dictionary) bool {
		return 0 < v && v < dic.IDFreq(id)
	})
}

func MinCount(v int) FilterFn {
	return FilterFn(func(id int, dic *dictionary.Dictionary) bool {
		return 0 <= v && dic.IDFreq(id) < v
	})
}
