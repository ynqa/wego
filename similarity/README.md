# Similarity

Calculate the cosine similarity between words.

## Usage

```
Estimate the similarity between words

Usage:
  word-embedding sim -i FILENAME WORD [flags]

Examples:
word-embedding sim -i example/word_vectors.txt apple

Flags:
  -i, --input string   Input path of a file written words' vector with libsvm format (default "example/word_vectors.txt")
  -r, --rank int       Set number of the similar word list displayed (default 10)
```

## Example

For instance, after running [demo](https://github.com/ynqa/word-embedding#demo). The results are not always the same.

```
$ go run main.go sim -i example/word_vectors_sg.txt microsoft
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
