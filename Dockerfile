FROM golang:1.9-alpine AS builder

ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64

WORKDIR /go/src/github.com/ynqa/word-embedding
COPY . .
RUN go build -o word-embedding .

FROM busybox
COPY --from=builder /go/src/github.com/ynqa/word-embedding/word-embedding /usr/local/bin/word-embedding

ENTRYPOINT ["word-embedding"]
CMD ["help"]
