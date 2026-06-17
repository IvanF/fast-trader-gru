FROM golang:1.22-alpine AS builder
WORKDIR /build
RUN apk add --no-cache git ca-certificates
COPY . .
RUN go mod tidy && go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /history_logger ./cmd/history_logger

FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata
COPY --from=builder /history_logger /usr/local/bin/history_logger
USER nobody
ENTRYPOINT ["/usr/local/bin/history_logger"]
