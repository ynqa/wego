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
    RANK |   WORD    |  COSINE
  +------+-----------+----------+
       1 | computers | 0.995368
       2 | ibm       | 0.993774
       3 | os        | 0.993721
       4 | machines  | 0.993713
       5 | operating | 0.993547
       6 | wikipedia | 0.993026
       7 | mpeg      | 0.992636
       8 | apple     | 0.992628
       9 | server    | 0.992574
      10 | unix      | 0.992385
```
