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

package model

// Model is the interface that has Train, Save.
type Model interface {
	// Train is function for
	Train() error
	Save(outputFile string) error
}

// SaveVectorType is a list of types to save model.
type SaveVectorType int

const (
	// NORMAL saves word vectors only.
	NORMAL SaveVectorType = iota
	// ADD add word to context vectors, and save them.
	ADD
)

func (t SaveVectorType) String() string {
	switch t {
	case NORMAL:
		return "normal"
	case ADD:
		return "add"
	default:
		return "unknown"
	}
}
