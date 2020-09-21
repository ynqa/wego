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

package repl

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"

	"github.com/peterh/liner"
	"github.com/pkg/errors"
	"github.com/ynqa/wego/pkg/embedding"
	"github.com/ynqa/wego/pkg/embedding/embutil"
	"github.com/ynqa/wego/pkg/search"
)

type searchparams struct {
	dim int
	k   int
}

type searchcursor struct {
	w1, w2 string
	vector []float64
}

type Repl struct {
	*liner.State
	searcher *search.Searcher
	cursor   *searchcursor
	params   *searchparams
}

func New(searcher *search.Searcher, k int) (*Repl, error) {
	if searcher.Items.Empty() {
		return nil, errors.New("Number of items for searcher must be over 0")
	}
	return &Repl{
		State:    liner.NewLiner(),
		searcher: searcher,
		cursor: &searchcursor{
			vector: make([]float64, searcher.Items[0].Dim),
		},
		params: &searchparams{
			dim: searcher.Items[0].Dim,
			k:   k,
		},
	}, nil
}

func (r *Repl) Run() error {
	defer r.Close()
	for {
		l, err := r.Prompt(">> ")
		if err != nil {
			fmt.Println("error: ", err)
		}
		switch l {
		case "exit":
			return nil
		case "":
			continue
		default:
			if err := r.eval(l); err != nil {
				fmt.Println(err)
			}
		}
	}
}

func (r *Repl) eval(l string) error {
	defer func() {
		r.cursor.w1 = ""
		r.cursor.w2 = ""
		r.cursor.vector = make([]float64, r.params.dim)
	}()

	expr, err := parser.ParseExpr(l)
	if err != nil {
		return err
	}

	var neighbors search.Neighbors
	switch e := expr.(type) {
	case *ast.Ident:
		neighbors, err = r.searcher.SearchInternal(e.String(), r.params.k)
		if err != nil {
			fmt.Printf("failed to search with word=%s\n", e.String())
		}
	case *ast.BinaryExpr:
		if err := r.evalExpr(expr); err != nil {
			return err
		}
		neighbors, err = r.searcher.Search(embedding.Embedding{
			Vector: r.cursor.vector,
			Norm:   embutil.Norm(r.cursor.vector),
		}, r.params.k, r.cursor.w1, r.cursor.w2)
		if err != nil {
			fmt.Printf("failed to search with vector=%v\n", r.cursor.vector)
		}
	default:
		return errors.Errorf("invalid type %v", e)
	}
	neighbors.Describe()
	return nil
}

func (r *Repl) evalExpr(expr ast.Expr) error {
	switch e := expr.(type) {
	case *ast.BinaryExpr:
		return r.evalBinaryExpr(e)
	case *ast.Ident:
		return nil
	default:
		return errors.Errorf("invalid type %v", e)
	}
}

func (r *Repl) evalBinaryExpr(expr *ast.BinaryExpr) error {
	xi, err := r.evalAsEmbedding(expr.X)
	if err != nil {
		return err
	}
	yi, err := r.evalAsEmbedding(expr.Y)
	if err != nil {
		return nil
	}
	r.cursor.w1 = xi.Word
	r.cursor.w2 = yi.Word
	r.cursor.vector, err = arithmetic(xi.Vector, expr.Op, yi.Vector)
	return err
}

func (r *Repl) evalAsEmbedding(expr ast.Expr) (embedding.Embedding, error) {
	if err := r.evalExpr(expr); err != nil {
		return embedding.Embedding{}, err
	}
	v, ok := expr.(*ast.Ident)
	if !ok {
		return embedding.Embedding{}, errors.Errorf("failed to parse %v", expr)
	}
	vi, ok := r.searcher.Items.Find(v.String())
	if !ok {
		return embedding.Embedding{}, errors.Errorf("not found word=%s in vector map", v.String())
	} else if err := vi.Validate(); err != nil {
		return embedding.Embedding{}, err
	}
	return vi, nil
}

func arithmetic(v1 []float64, op token.Token, v2 []float64) ([]float64, error) {
	switch op {
	case token.ADD:
		return add(v1, v2)
	case token.SUB:
		return sub(v1, v2)
	default:
		return nil, errors.Errorf("invalid operator %v", op.String())
	}
}
