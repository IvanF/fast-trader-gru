#!/bin/sh
set -eu

INFLUX_HOST="${INFLUX_HOST:-http://influxdb:8086}"
INFLUX_ORG="${INFLUX_ORG:-fasttrader}"
INFLUX_TOKEN="${INFLUX_TOKEN:?INFLUX_TOKEN required}"

echo "Waiting for InfluxDB at ${INFLUX_HOST}..."
until influx ping --host "${INFLUX_HOST}" >/dev/null 2>&1; do
  sleep 2
done

echo "Creating aggregated features bucket (365d retention)..."
influx bucket create \
  --host "${INFLUX_HOST}" \
  --org "${INFLUX_ORG}" \
  --token "${INFLUX_TOKEN}" \
  --name market_features \
  --retention 8760h \
  2>/dev/null || echo "bucket market_features already exists"

echo "Removing duplicate downsample tasks (if any)..."
influx task list --host "${INFLUX_HOST}" --token "${INFLUX_TOKEN}" --hide-headers 2>/dev/null \
  | while read -r id _rest; do
      [ -n "${id}" ] || continue
      influx task delete --host "${INFLUX_HOST}" --token "${INFLUX_TOKEN}" --id "${id}" 2>/dev/null || true
    done

echo "Registering downsample task (1m aggregates -> market_features)..."
influx task create \
  --host "${INFLUX_HOST}" \
  --org "${INFLUX_ORG}" \
  --token "${INFLUX_TOKEN}" \
  --file /init/downsample_1m.flux

echo "InfluxDB init complete."
