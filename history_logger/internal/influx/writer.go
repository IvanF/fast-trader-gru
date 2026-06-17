package influx

import (
	"fmt"
	"log/slog"
	"sync"
	"time"

	influxdb2 "github.com/influxdata/influxdb-client-go/v2"
	"github.com/influxdata/influxdb-client-go/v2/api"

	"github.com/fast-trader-gru/history_logger/internal/config"
	"github.com/fast-trader-gru/history_logger/internal/metrics"
)

type BatchWriter struct {
	cfg      config.Config
	writeAPI api.WriteAPI
	logger   *slog.Logger

	incoming chan string
	done     chan struct{}
	wg       sync.WaitGroup

	mu          sync.Mutex
	buffer      []string
	lastInfoLog time.Time
}

func NewBatchWriter(cfg config.Config, logger *slog.Logger) (*BatchWriter, error) {
	if cfg.InfluxToken == "" {
		return nil, fmt.Errorf("INFLUX_TOKEN is required")
	}
	client := influxdb2.NewClient(cfg.InfluxURL, cfg.InfluxToken)
	writeAPI := client.WriteAPI(cfg.InfluxOrg, cfg.InfluxBucket)

	w := &BatchWriter{
		cfg:      cfg,
		writeAPI: writeAPI,
		logger:   logger,
		incoming: make(chan string, cfg.PointBufferSize),
		done:     make(chan struct{}),
		buffer:   make([]string, 0, cfg.BatchMaxPoints),
	}

	w.wg.Add(1)
	go w.flushLoop()

	go func() {
		for err := range writeAPI.Errors() {
			metrics.InfluxWriteErrors.Inc()
			logger.Error("influx async write error", "error", err)
		}
	}()

	return w, nil
}

// Enqueue is non-blocking. Drops and counts when the buffer channel is full.
func (w *BatchWriter) Enqueue(line string) {
	select {
	case w.incoming <- line:
	default:
		metrics.PointsDropped.Inc()
	}
}

func (w *BatchWriter) flushLoop() {
	defer w.wg.Done()
	ticker := time.NewTicker(w.cfg.BatchFlushEvery)
	defer ticker.Stop()

	for {
		select {
		case <-w.done:
			w.drainIncoming()
			w.flush()
			return
		case line := <-w.incoming:
			w.append(line)
			if w.len() >= w.cfg.BatchMaxPoints {
				w.flush()
			}
		case <-ticker.C:
			if w.len() > 0 {
				w.flush()
			}
		}
	}
}

func (w *BatchWriter) drainIncoming() {
	for {
		select {
		case line := <-w.incoming:
			w.append(line)
		default:
			return
		}
	}
}

func (w *BatchWriter) append(line string) {
	w.mu.Lock()
	w.buffer = append(w.buffer, line)
	metrics.BufferDepth.Set(float64(len(w.buffer)))
	w.mu.Unlock()
}

func (w *BatchWriter) len() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.buffer)
}

func (w *BatchWriter) flush() {
	w.mu.Lock()
	if len(w.buffer) == 0 {
		w.mu.Unlock()
		return
	}
	batch := w.buffer
	w.buffer = make([]string, 0, w.cfg.BatchMaxPoints)
	metrics.BufferDepth.Set(0)
	w.mu.Unlock()

	start := time.Now()
	for _, line := range batch {
		w.writeAPI.WriteRecord(line)
	}
	w.writeAPI.Flush()

	metrics.PointsWritten.Add(float64(len(batch)))
	metrics.BatchFlushes.Inc()
	metrics.FlushDuration.Observe(time.Since(start).Seconds())
	metrics.BatchSize.Observe(float64(len(batch)))
	if time.Since(w.lastInfoLog) >= 60*time.Second {
		w.lastInfoLog = time.Now()
		w.logger.Info("influx batch flushed",
			"points", len(batch),
			"duration_ms", time.Since(start).Milliseconds(),
		)
	} else {
		w.logger.Debug("flushed batch", "points", len(batch), "duration_ms", time.Since(start).Milliseconds())
	}
}

func (w *BatchWriter) Close() {
	close(w.done)
	w.wg.Wait()
	w.writeAPI.Flush()
}
