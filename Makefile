TAG = latest

.PHONY: build
build:
	go build

.PHONY: clean
clean:
	rm -rf vendor/

.PHONY: docker-build
docker-build:
	docker build -t wego:${TAG} .

.PHONY: ensure
ensure:
	dep ensure -vendor-only

.PHONY: fmt
fmt:
	go fmt `go list ./...`

.PHONY: lint
lint:
	golint `go list ./...`

.PHONY: test
test:
	go test -cover -v `go list ./...`
