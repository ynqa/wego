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

const gloveFlagSize = 6

func TestGloveBind(t *testing.T) {
	defer viper.Reset()

	gloveBind(GloveCmd)

	if len(viper.AllKeys()) != gloveFlagSize {
		t.Errorf("Expected similarityBind maps %v keys: %v",
			gloveFlagSize, viper.AllKeys())
	}
}

func TestGloveCmdPreRun(t *testing.T) {
	defer viper.Reset()

	var empty []string
	GloveCmd.PreRun(Word2VecCmd, empty)

	if len(viper.AllKeys()) != gloveFlagSize+configFlagSize {
		t.Errorf("Expected PreRun of Word2VecCmd maps %v keys: %v",
			gloveFlagSize+configFlagSize, viper.AllKeys())
	}
}
