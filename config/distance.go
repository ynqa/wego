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

package config

// DistanceConfig is enum of the common config.
type DistanceConfig int

// The list of DistanceConfig.
const (
	Rank DistanceConfig = iota
)

// The defaults of DistanceConfig.
const (
	DefaultRank int = 10
)

func (d DistanceConfig) String() string {
	switch d {
	case Rank:
		return "rank"
	default:
		return "unknown"
	}
}
