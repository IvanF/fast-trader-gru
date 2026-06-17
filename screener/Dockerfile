FROM golang:1.22-alpine AS builder
WORKDIR /build
RUN apk add --no-cache git ca-certificates
COPY . .
RUN go mod tidy && go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /screener ./cmd/screener

FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata
COPY --from=builder /screener /usr/local/bin/screener
USER nobody
ENTRYPOINT ["/usr/local/bin/screener"]
