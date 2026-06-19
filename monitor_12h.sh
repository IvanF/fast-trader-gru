#!/bin/bash
# 12-hour trading monitor with auto-optimization
LOG="/media/i/48c2d5df-6beb-4ce0-9fdd-54a72017069d1/bigrepo/crypto/lstm/fast-trader-gru"
REPORT="$LOG/trading_report_12h.txt"
OPTIMIZATIONS="$LOG/optimization_log.txt"
ENV_FILE="$LOG/.env"

echo "=== 12H MONITOR STARTED: $(date) ===" > "$REPORT"

while true; do
    echo "" >> "$REPORT"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$REPORT"
    
    # 1. Trade performance
    docker logs ftg-oms 2>&1 | grep '"position closed"' | python3 -c "
import sys, json
trades = []
for line in sys.stdin:
    try: trades.append(json.loads(line.strip()))
    except: pass

total_pnl = sum(t.get('pnl', 0) for t in trades)
wins = [t for t in trades if t.get('pnl', 0) > 0]
losses = [t for t in trades if t.get('pnl', 0) <= 0]
total = len(trades)

print(f'Trades: {total} | WR: {len(wins)/total*100:.1f}% | PnL: \${total_pnl:.4f}')

if total > 0:
    avg_win = sum(t['pnl'] for t in wins)/len(wins) if wins else 0
    avg_loss = sum(t['pnl'] for t in losses)/len(losses) if losses else 0
    print(f'Avg win: \${avg_win:.4f} | Avg loss: \${avg_loss:.4f}')
    print(f'W/L ratio: {abs(avg_win/avg_loss):.2f}' if avg_loss else '')
    
    reasons = {}
    for t in trades: r = t.get('reason','?'); reasons[r] = reasons.get(r,0)+1
    print(f'Reasons: {reasons}')
    
    # Worst symbols
    by_sym = {}
    for t in trades:
        s = t.get('symbol','?')
        by_sym[s] = by_sym.get(s, {'pnl':0, 'count':0})
        by_sym[s]['pnl'] += t.get('pnl', 0)
        by_sym[s]['count'] += 1
    worst = sorted(by_sym.items(), key=lambda x: x[1]['pnl'])[:3]
    print('Worst symbols:', [(s, f\"\\\${d['pnl']:.4f}\") for s, d in worst])
" >> "$REPORT" 2>/dev/null
    
    # 2. Check for optimization needed
    docker logs ftg-oms 2>&1 | grep '"position closed"' | python3 -c "
import sys, json
trades = []
for line in sys.stdin:
    try: trades.append(json.loads(line.strip()))
    except: pass

total_pnl = sum(t.get('pnl', 0) for t in trades)
wins = [t for t in trades if t.get('pnl', 0) > 0]
total = len(trades)

if total >= 5:
    wr = len(wins)/total*100
    avg_win = sum(t['pnl'] for t in wins)/len(wins) if wins else 0
    losses = [t for t in trades if t.get('pnl', 0) <= 0]
    avg_loss = sum(t['pnl'] for t in losses)/len(losses) if losses else 0
    
    # Auto-optimize: if WR < 40% or total PnL negative
    needs_optimize = False
    reasons = []
    if wr < 40:
        needs_optimize = True
        reasons.append(f'low_wr={wr:.0f}%')
    if total_pnl < -5:
        needs_optimize = True
        reasons.append(f'high_loss={total_pnl:.2f}')
    if avg_loss != 0 and abs(avg_win/avg_loss) < 1.0:
        needs_optimize = True
        reasons.append(f'bad_wl_ratio={abs(avg_win/avg_loss):.2f}')
    
    if needs_optimize:
        print(f'OPTIMIZE_NEEDED: {reasons}')
    else:
        print('SYSTEM_HEALTHY')
else:
    print(f'INSUFFICIENT_DATA: {total} trades')
" >> "$REPORT" 2>/dev/null
    
    echo "" >> "$REPORT"
    sleep 1800  # 30 minutes
done
