// Copyright © 2019 Makoto Ito
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
	"io"

	"github.com/peterh/liner"
	"github.com/pkg/errors"
	"github.com/ynqa/wego/pkg/search"
)

var (
	vector []float64
)

type searchParam struct {
	dim int
	k   int
}

type Repl struct {
	*liner.State
	searcher *search.Searcher
	param    *searchParam
}

func New(r io.Reader, k int) (*Repl, error) {
	searcher, err := search.NewForVectorFile(r)
	if err != nil {
		return nil, err
	}
	if searcher.Items.Empty() {
		return nil, errors.New("Number of items for searcher must be over 0")
	}
	return &Repl{
		State:    liner.NewLiner(),
		searcher: searcher,
		param: &searchParam{
			dim: len(searcher.Items),
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
		vector = make([]float64, r.param.dim)
	}()

	expr, err := parser.ParseExpr(l)
	if err != nil {
		return err
	}

	var neighbors search.Neighbors
	switch e := expr.(type) {
	case *ast.Ident:
		neighbors, err = r.searcher.InternalSearch(e.String(), r.param.k)
		if err != nil {
			fmt.Printf("failed to search with word=%s\n", e.String())
		}
	case *ast.BinaryExpr:
		if err := r.evalExpr(expr); err != nil {
			return err
		}
		neighbors, err = r.searcher.Search(vector, r.param.k)
		if err != nil {
			fmt.Printf("failed to search with vector=%v\n", vector)
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
	xi, err := r.evalAsItem(expr.X)
	if err != nil {
		return err
	}
	yi, err := r.evalAsItem(expr.Y)
	if err != nil {
		return nil
	}
	vector, err = arithmetic(xi.Vector, expr.Op, yi.Vector)
	return err
}

func (r *Repl) evalAsItem(expr ast.Expr) (search.Item, error) {
	if err := r.evalExpr(expr); err != nil {
		return search.Item{}, err
	}
	v, ok := expr.(*ast.Ident)
	if !ok {
		return search.Item{}, errors.Errorf("failed to parse %v", expr)
	}
	vi, ok := r.searcher.Items.Find(v.String())
	if !ok {
		return search.Item{}, errors.Errorf("not found word=%s in vector map", v.String())
	} else if err := vi.Validate(); err != nil {
		return search.Item{}, err
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
