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

package search

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/embedding/embutil"
)

func TestSearchInternal(t *testing.T) {
	type args struct {
		word string
		k    int
	}

	testCases := []struct {
		name   string
		items  embedding.Embeddings
		args   args
		expect Neighbors
	}{
		{
			name: "search internal",
			items: embedding.Embeddings{
				{
					Word:   "apple",
					Dim:    5,
					Vector: []float64{1, 1, 1, 1, 1},
					Norm:   embutil.Norm([]float64{1, 1, 1, 1, 1}),
				},
				{
					Word:   "banana",
					Dim:    5,
					Vector: []float64{1, 1, 1, 1, 1},
					Norm:   embutil.Norm([]float64{1, 1, 1, 1, 1}),
				},
				{
					Word:   "chocolate",
					Dim:    5,
					Vector: []float64{0, 0, 0, 0, 0},
					Norm:   embutil.Norm([]float64{0, 0, 0, 0, 0}),
				},
				{
					Word:   "dragon",
					Dim:    5,
					Vector: []float64{-1, -1, -1, -1, -1},
					Norm:   embutil.Norm([]float64{-1, -1, -1, -1, -1}),
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
			neighbors, _ := s.SearchInternal(tc.args.word, tc.args.k)
			assert.Truef(t, reflect.DeepEqual(neighbors, tc.expect), "Must be equal %v and %v", neighbors, tc.expect)
		})
	}
}

func TestSearchVector(t *testing.T) {
	type args struct {
		query []float64
		k     int
	}

	testCases := []struct {
		name   string
		items  embedding.Embeddings
		args   args
		expect Neighbors
	}{
		{
			name: "search vector",
			items: embedding.Embeddings{
				{
					Word:   "apple",
					Dim:    5,
					Vector: []float64{1, 1, 1, 1, 1},
					Norm:   embutil.Norm([]float64{1, 1, 1, 1, 1}),
				},
				{
					Word:   "banana",
					Dim:    5,
					Vector: []float64{1, 1, 1, 1, 1},
					Norm:   embutil.Norm([]float64{1, 1, 1, 1, 1}),
				},
				{
					Word:   "chocolate",
					Dim:    5,
					Vector: []float64{0, 0, 0, 0, 0},
					Norm:   embutil.Norm([]float64{0, 0, 0, 0, 0}),
				},
				{
					Word:   "dragon",
					Dim:    5,
					Vector: []float64{-1, -1, -1, -1, -1},
					Norm:   embutil.Norm([]float64{-1, -1, -1, -1, -1}),
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
			neighbors, _ := s.SearchVector(tc.args.query, tc.args.k)
			assert.Truef(t, reflect.DeepEqual(tc.expect, neighbors), "Must be equal %v and %v", tc.expect, neighbors)
		})
	}
}
