.PHONY: build
build:
	go build

.PHONY: clean
clean:
	rm -rf vendor/

.PHONY: fmt
fmt:
	go fmt `glide nv`

.PHONY: lint
lint:
	golint `glide nv`

.PHONY: test
test:
	go test -cover -v `glide nv`