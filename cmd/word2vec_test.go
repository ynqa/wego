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

package cmd

import (
	"testing"

	"github.com/spf13/viper"
)

const word2vecFlagSize = 7

func TestWord2vecBind(t *testing.T) {
	defer viper.Reset()

	word2vecBind(Word2vecCmd)

	if len(viper.AllKeys()) != word2vecFlagSize {
		t.Errorf("Expected word2vecBind maps %v keys: %v",
			word2vecFlagSize,
			viper.AllKeys())
	}
}

func TestWord2vecCmdPreRun(t *testing.T) {
	defer viper.Reset()

	var empty []string
	Word2vecCmd.PreRun(Word2vecCmd, empty)

	if len(viper.AllKeys()) != word2vecFlagSize+configFlagSize {
		t.Errorf("Expected PreRun of Word2vecCmd maps %v keys: %v",
			word2vecFlagSize+configFlagSize, viper.AllKeys())
	}
}
