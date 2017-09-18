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

package word2vec

import (
	"bytes"
	"io"
	"io/ioutil"
	"testing"

	"github.com/ynqa/word-embedding/config"
	"github.com/ynqa/word-embedding/model"
)

var mockConfig = model.NewConfig(
	config.DefaultLower,
	config.DefaultDimension,
	config.DefaultWindow,
	config.DefaultInitLearningRate,
)

var mockState = NewState(
	mockConfig,
	mockOpt,
	config.DefaultSubsampleThreshold,
	config.DefaultTheta,
	config.DefaultBatchSize,
)

var mockText = "A A A A B B B B C C C C"

type nopSeeker struct {
	io.ReadCloser
}

func (n nopSeeker) Seek(offset int64, whence int) (int64, error) {
	return 0, nil
}

func TestPreprocess(t *testing.T) {
	f := nopSeeker{
		ReadCloser: ioutil.NopCloser(bytes.NewReader([]byte(mockText))),
	}

	if _, err := mockState.Preprocess(f); err != nil {
		t.Error("Word2Vec: Preprocess returns error")
	}
}
