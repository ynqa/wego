FROM golang:1.10.3-alpine3.8 AS builder

RUN apk update \
  && apk add --no-cache git \
  && go get -u github.com/golang/dep/cmd/dep

ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64

WORKDIR /go/src/github.com/ynqa/wego
COPY . .
RUN dep ensure -v -vendor-only
RUN go build -v -o wego .

FROM busybox
COPY --from=builder /go/src/github.com/ynqa/wego/wego /usr/local/bin/wego

ENTRYPOINT ["wego"]
CMD ["help"]
