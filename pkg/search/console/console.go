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

package console

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

type Console struct {
	*liner.State
	searcher *search.Searcher
	cursor   *searchcursor
	params   *searchparams
}

func New(searcher *search.Searcher, k int) (*Console, error) {
	if searcher.Items.Empty() {
		return nil, errors.New("Number of items for searcher must be over 0")
	}
	return &Console{
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

func (c *Console) Run() error {
	defer c.Close()
	for {
		l, err := c.Prompt(">> ")
		if err != nil {
			fmt.Println("error: ", err)
		}
		switch l {
		case "exit":
			return nil
		case "":
			continue
		default:
			if err := c.eval(l); err != nil {
				fmt.Println(err)
			}
		}
	}
}

func (c *Console) eval(l string) error {
	defer func() {
		c.cursor.w1 = ""
		c.cursor.w2 = ""
		c.cursor.vector = make([]float64, c.params.dim)
	}()

	expr, err := parser.ParseExpr(l)
	if err != nil {
		return err
	}

	var neighbors search.Neighbors
	switch e := expr.(type) {
	case *ast.Ident:
		neighbors, err = c.searcher.SearchInternal(e.String(), c.params.k)
		if err != nil {
			fmt.Printf("failed to search with word=%s\n", e.String())
		}
	case *ast.BinaryExpr:
		if err := c.evalExpr(expr); err != nil {
			return err
		}
		neighbors, err = c.searcher.Search(embedding.Embedding{
			Vector: c.cursor.vector,
			Norm:   embutil.Norm(c.cursor.vector),
		}, c.params.k, c.cursor.w1, c.cursor.w2)
		if err != nil {
			fmt.Printf("failed to search with vector=%v\n", c.cursor.vector)
		}
	default:
		return errors.Errorf("invalid type %v", e)
	}
	neighbors.Describe()
	return nil
}

func (c *Console) evalExpr(expr ast.Expr) error {
	switch e := expr.(type) {
	case *ast.BinaryExpr:
		return c.evalBinaryExpr(e)
	case *ast.Ident:
		return nil
	default:
		return errors.Errorf("invalid type %v", e)
	}
}

func (c *Console) evalBinaryExpr(expr *ast.BinaryExpr) error {
	xi, err := c.evalAsEmbedding(expr.X)
	if err != nil {
		return err
	}
	yi, err := c.evalAsEmbedding(expr.Y)
	if err != nil {
		return nil
	}
	c.cursor.w1 = xi.Word
	c.cursor.w2 = yi.Word
	c.cursor.vector, err = arithmetic(xi.Vector, expr.Op, yi.Vector)
	return err
}

func (c *Console) evalAsEmbedding(expr ast.Expr) (embedding.Embedding, error) {
	if err := c.evalExpr(expr); err != nil {
		return embedding.Embedding{}, err
	}
	v, ok := expr.(*ast.Ident)
	if !ok {
		return embedding.Embedding{}, errors.Errorf("failed to parse %v", expr)
	}
	vi, ok := c.searcher.Items.Find(v.String())
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
