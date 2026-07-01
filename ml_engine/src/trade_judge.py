"""Trade Judge — классификатор исходов сделок.

Классифицирует каждую закрытую сделку по причине результата:
  - OPTIMAL: сделка закрыта по TP, R:R >= 1:1.5
  - UNDER_OPTIMIZED_TP: MFE > TP * 1.5 (могли заработать больше)
  - BAD_ENTRY_SIGNAL: MAE > TP * 2 (сразу пошли против нас)
  - TOXIC_PATTERN: MAE >> MFE, паттерн неспасаемый
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


TRADE_JUDGES = {
    "OPTIMAL": "S TP закрыта, R:R адекватен",
    "UNDER_OPTIMIZED_TP": "MFE >> TP — ставь TP выше",
    "BAD_ENTRY_SIGNAL": "MAE >> TP — вход был плохим",
    "TOXIC_PATTERN": "Паттерн неспасаемый",
    "UNCERTAIN": "Недостаточно данных для вердикта",
}


@dataclass
class TradeVerdict:
    judge: str
    confidence: float
    optimal_sl_pct: float
    optimal_tp_pct: float
    reason: str
    mae_pct: float = 0.0
    mfe_pct: float = 0.0


def classify_trade(
    entry_price: float,
    exit_price: float,
    direction: str,
    pnl: float,
    close_reason: str,
    mae_pct: float = 0.0,
    mfe_pct: float = 0.0,
    min_salvage_rr: float = 1.5,
) -> TradeVerdict:
    """Классифицирует сделку по MAE/MFE и результату."""

    if abs(entry_price) < 1e-10:
        return TradeVerdict("UNCERTAIN", 0.0, 0.0, 0.0, "no entry price")

    if close_reason.startswith("shadow_"):
        close_reason = close_reason[len("shadow_"):]

    if close_reason == "take_profit":
        abs_mae = abs(mae_pct)
        abs_mfe = abs(mfe_pct)
        opt_sl = abs_mae * 1.1
        opt_tp = abs_mfe * 0.9

        if abs_mfe > 0 and abs_mae > 0:
            actual_rr = abs_mfe / abs_mae
        elif abs_mae < 0.001:
            actual_rr = 999.0
        else:
            actual_rr = 0.0

        if actual_rr >= min_salvage_rr and pnl >= 0:
            return TradeVerdict(
                judge="OPTIMAL",
                confidence=min(actual_rr / 3.0, 1.0),
                optimal_sl_pct=opt_sl,
                optimal_tp_pct=opt_tp,
                reason=f"TP hit, R:R={actual_rr:.1f}",
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
            )

        if abs_mfe > opt_tp * 1.5:
            return TradeVerdict(
                judge="UNDER_OPTIMIZED_TP",
                confidence=min(abs_mfe / (opt_tp + 0.001) * 0.3, 1.0),
                optimal_sl_pct=opt_sl,
                optimal_tp_pct=opt_tp,
                reason=f"MFE={abs_mfe:.2%} >> TP={opt_tp:.2%}, ставь TP выше",
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
            )

        return TradeVerdict(
            judge="OPTIMAL",
            confidence=0.7,
            optimal_sl_pct=opt_sl,
            optimal_tp_pct=opt_tp,
            reason=f"TP hit, R:R={actual_rr:.1f}",
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
        )

    if close_reason == "stop_loss":
        abs_mae = abs(mae_pct)
        abs_mfe = abs(mfe_pct)
        opt_sl = abs_mae * 1.1
        opt_tp = abs_mfe * 0.9 if abs_mfe > 0 else abs_mae * 2.0

        if abs_mae > 0 and abs_mfe > 0:
            salvage_rr = abs_mfe / abs_mae
        else:
            salvage_rr = 0.0

        if abs_mae > 0.02 and (abs_mfe < 0.005 or salvage_rr < 0.5):
            return TradeVerdict(
                judge="TOXIC_PATTERN",
                confidence=min(abs_mae / 0.05, 1.0),
                optimal_sl_pct=opt_sl,
                optimal_tp_pct=opt_tp,
                reason=f"MAE={abs_mae:.2%} >> MFE={abs_mfe:.2%}, неспасаемый",
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
            )

        if abs_mae > 0.005 and salvage_rr >= min_salvage_rr:
            return TradeVerdict(
                judge="UNDER_OPTIMIZED_TP",
                confidence=0.6,
                optimal_sl_pct=opt_sl,
                optimal_tp_pct=opt_tp,
                reason=f"SL сработал, но MFE={abs_mfe:.2%} был достаточным, нужен более широкий SL",
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
            )

        return TradeVerdict(
            judge="BAD_ENTRY_SIGNAL",
            confidence=min(abs_mae / 0.02, 1.0),
            optimal_sl_pct=opt_sl,
            optimal_tp_pct=opt_tp,
            reason=f"MAE={abs_mae:.2%}, вход был неудачным",
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
        )

    abs_mae = abs(mae_pct)
    abs_mfe = abs(mfe_pct)
    opt_sl = abs_mae * 1.1 if abs_mae > 0 else 0.005
    opt_tp = abs_mfe * 0.9 if abs_mfe > 0 else 0.008

    if pnl >= 0:
        return TradeVerdict(
            judge="OPTIMAL",
            confidence=0.5,
            optimal_sl_pct=opt_sl,
            optimal_tp_pct=opt_tp,
            reason=f"closed {close_reason} profitable",
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
        )

    return TradeVerdict(
        judge="BAD_ENTRY_SIGNAL",
        confidence=0.4,
        optimal_sl_pct=opt_sl,
        optimal_tp_pct=opt_tp,
        reason=f"closed {close_reason} loss",
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
    )
