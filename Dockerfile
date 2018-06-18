FROM golang:1.9-alpine AS builder

ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64

WORKDIR /go/src/github.com/ynqa/wego
COPY . .
RUN go build -o wego .

FROM busybox
COPY --from=builder /go/src/github.com/ynqa/wego/wego /usr/local/bin/wego

ENTRYPOINT ["wego"]
CMD ["help"]
