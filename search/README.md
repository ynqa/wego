# Search

Similarity search between word vectors.

## Usage

```
Search similar words

Usage:
  wego search [flags]

Examples:
  wego search -i example/word_vectors.txt microsoft

Flags:
  -h, --help               help for search
  -i, --inputFile string   input file path for trained word vector (default "example/input.txt")
  -r, --rank int           how many the most similar words will be displayed (default 10)
```

## Example

```
$ go run wego.go search -i example/word_vectors_sg.txt microsoft
  RANK |    WORD    | SIMILARITY
+------+------------+------------+
     1 | apple      |   0.994008
     2 | operating  |   0.992855
     3 | versions   |   0.992800
     4 | ibm        |   0.992232
     5 | os         |   0.989174
     6 | computers  |   0.988998
     7 | machines   |   0.988804
     8 | dvd        |   0.988732
     9 | cd         |   0.988259
    10 | compatible |   0.988200
```
