.PHONY: build
build:
	go build

.PHONY: clean-vendor
clean-vendor:
	rm -rf vendor/

.PHONY: clean-example
clean-example:
	rm -rf example/*.txt

.PHONY: clean
clean: clean-example clean-vendor

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
