option task = {
    name: "downsample_market_1m",
    every: 1m,
}

numericFields = ["price", "size", "bid_vol", "ask_vol", "obi", "levels"]

from(bucket: "market_raw")
    |> range(start: -2m)
    |> filter(fn: (r) => r._measurement == "orderbook_summary" or r._measurement == "trades")
    |> filter(fn: (r) => contains(value: r._field, set: numericFields))
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    |> to(bucket: "market_features", org: "fasttrader")
