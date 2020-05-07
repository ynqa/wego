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

package search

import (
	"bytes"
	"io/ioutil"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ynqa/wego/pkg/item"
	"github.com/ynqa/wego/pkg/search/searchutil"
)

func TestNewForVectorFile(t *testing.T) {
	testCases := []struct {
		name     string
		contents string
		itemSize int
	}{
		{
			name: "read vector file",
			contents: `apple 1 1 1 1 1
			banana 1 1 1 1 1
			chocolate 0 0 0 0 0
			dragon -1 -1 -1 -1 -1`,
			itemSize: 4,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r := ioutil.NopCloser(bytes.NewReader([]byte(tc.contents)))
			defer r.Close()

			s, _ := NewForVectorFile(r)
			assert.Equal(t, tc.itemSize, len(s.Items))
		})
	}
}

func TestInternalSearch(t *testing.T) {
	type args struct {
		word string
		k    int
	}

	testCases := []struct {
		name   string
		items  []item.Item
		args   args
		expect Neighbors
	}{
		{
			name: "internal search",
			items: []item.Item{
				{
					Word:   "apple",
					Dim:    5,
					Vector: []float64{1, 1, 1, 1, 1},
				},
				{
					Word:   "banana",
					Dim:    5,
					Vector: []float64{1, 1, 1, 1, 1},
				},
				{
					Word:   "chocolate",
					Dim:    5,
					Vector: []float64{0, 0, 0, 0, 0},
				},
				{
					Word:   "dragon",
					Dim:    5,
					Vector: []float64{-1, -1, -1, -1, -1},
				},
			},
			args: args{
				word: "apple",
				k:    1,
			},
			expect: Neighbors{
				{
					Word:       "banana",
					Rank:       1,
					Similarity: 1.,
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			s, _ := New(tc.items...)
			neighbors, _ := s.InternalSearch(tc.args.word, tc.args.k)
			assert.Truef(t, reflect.DeepEqual(neighbors, tc.expect), "Must be equal %v and %v", neighbors, tc.expect)
		})
	}
}

func TestSearch(t *testing.T) {
	type args struct {
		query []float64
		k     int
	}

	testCases := []struct {
		name   string
		items  []item.Item
		args   args
		expect Neighbors
	}{
		{
			name: "internal search",
			items: []item.Item{
				{
					Word:   "apple",
					Dim:    5,
					Vector: []float64{1, 1, 1, 1, 1},
				},
				{
					Word:   "banana",
					Dim:    5,
					Vector: []float64{1, 1, 1, 1, 1},
				},
				{
					Word:   "chocolate",
					Dim:    5,
					Vector: []float64{0, 0, 0, 0, 0},
				},
				{
					Word:   "dragon",
					Dim:    5,
					Vector: []float64{-1, -1, -1, -1, -1},
				},
			},
			args: args{
				query: []float64{-1, -1, -1, -1, -1},
				k:     1,
			},
			expect: Neighbors{
				{
					Word:       "dragon",
					Rank:       1,
					Similarity: 1.,
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			s, _ := New(tc.items...)
			neighbors, _ := s.Search(tc.args.query, tc.args.k)
			assert.Truef(t, reflect.DeepEqual(tc.expect, neighbors), "Must be equal %v and %v", tc.expect, neighbors)
		})
	}
}

func TestNorm(t *testing.T) {
	testCases := []struct {
		name   string
		vec    []float64
		expect float64
	}{
		{
			name:   "norm",
			vec:    []float64{1, 1, 1, 1, 0, 0},
			expect: 2.,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expect, searchutil.Norm(tc.vec))
		})
	}
}

func TestCosine(t *testing.T) {
	testCases := []struct {
		name   string
		v1     []float64
		v2     []float64
		expect float64
	}{
		{
			name:   "cosine",
			v1:     []float64{1, 1, 1, 1, 0, 0},
			v2:     []float64{1, 1, 0, 0, 1, 1},
			expect: 0.5,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expect, searchutil.Cosine(tc.v1, tc.v2, searchutil.Norm(tc.v1), searchutil.Norm(tc.v2)))
		})
	}
}
