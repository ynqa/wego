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

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

const configFlagSize = 11

func TestConfigFlagSet(t *testing.T) {
	fs := ConfigFlagSet()

	if !fs.HasAvailableFlags() {
		t.Error("Expected that ConfigFlagSet() retruns *pflag.FlagSet without empty")
	}
}

func TestConfigBind(t *testing.T) {
	defer viper.Reset()

	config := &cobra.Command{}
	config.Flags().AddFlagSet(ConfigFlagSet())
	configBind(config)

	if len(viper.AllKeys()) != configFlagSize {
		t.Errorf("Expected configBind maps %v keys: %v",
			configFlagSize, viper.AllKeys())
	}
}
