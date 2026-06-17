module github.com/fast-trader-gru/data_ingestion

go 1.22

require (
	github.com/gorilla/websocket v1.5.1
	github.com/prometheus/client_golang v1.19.0
	github.com/redis/go-redis/v9 v9.3.0
	github.com/vmihailenco/msgpack/v5 v5.4.1
)

replace github.com/munnerz/goautoneg => github.com/munnerz/goautoneg v0.0.0-20180727004023-cc286d45f135
