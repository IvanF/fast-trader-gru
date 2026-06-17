package influx

import (
	"context"

	influxdb2 "github.com/influxdata/influxdb-client-go/v2"

	"github.com/fast-trader-gru/history_logger/internal/config"
)

func HealthCheck(ctx context.Context, cfg config.Config) error {
	client := influxdb2.NewClient(cfg.InfluxURL, cfg.InfluxToken)
	defer client.Close()
	_, err := client.Health(ctx)
	return err
}
