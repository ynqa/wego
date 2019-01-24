# REPL

REPL mode for similarity search

![wego](https://user-images.githubusercontent.com/6745370/51677211-2e54e700-201c-11e9-8ce9-19d4b84ef071.gif)

## Usage

```
Search similar words with REPL mode

Usage:
  wego repl [flags]

Examples:
  wego repl -i example/word_vectors.txt
  >> apple + banana
  ...

Flags:
  -h, --help               help for repl
  -i, --inputFile string   input file path for trained word vector (default "example/word_vectors.txt")
  -r, --rank int           how many the most similar words will be displayed (default 10)
```

## Example

Now, it is able to use `+`, `-` for arithmetic operations.

```
$ go run wego.go repl -i example/word_vectors_sg.txt
>> a + b
  RANK |  WORD   | SIMILARITY
+------+---------+------------+
     1 | phi     |   0.907975
     2 | q       |   0.904593
     3 | mathbf  |   0.903066
     4 | cdot    |   0.902205
     5 | b       |   0.901952
     6 | becomes |   0.900346
     7 | int     |   0.898680
     8 | z       |   0.897895
     9 | named   |   0.896480
    10 | v       |   0.895456
```
