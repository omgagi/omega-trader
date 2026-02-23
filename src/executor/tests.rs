use super::*;
use crate::execution::{ExecutionPlan, ImmediatePlan, Side};
use chrono::Utc;

#[test]
fn test_circuit_breaker_defaults() {
    let cb = CircuitBreaker::default();
    assert!((cb.max_deviation_pct - 0.02).abs() < 1e-6);
    assert_eq!(cb.max_consecutive_failures, 3);
}

#[test]
fn test_state_persistence_roundtrip() {
    let plan = ExecutionPlan::Immediate(ImmediatePlan {
        symbol: "AAPL".into(),
        side: Side::Buy,
        quantity: 10.0,
        estimated_price: 150.0,
        estimated_usd: 1500.0,
    });
    let state = ExecutionState::new(&plan);
    let json = persist_state(&state).unwrap();
    let recovered = recover_state(&json).unwrap();
    assert_eq!(recovered.status, ExecutionStatus::Pending);
    assert_eq!(recovered.total_slices, 1);
}

#[test]
fn test_daily_limits_check() {
    let limits = DailyLimits::new(10, 10_000.0, 5);
    assert!(limits.check(500.0).is_ok());
}

#[test]
fn test_daily_limits_exceeded_trades() {
    let mut limits = DailyLimits::new(2, 100_000.0, 0);
    limits.trades_today = 2;
    assert!(limits.check(100.0).is_err());
}

#[test]
fn test_daily_limits_exceeded_usd() {
    let mut limits = DailyLimits::new(100, 1_000.0, 0);
    limits.usd_today = 900.0;
    assert!(limits.check(200.0).is_err());
}

#[test]
fn test_daily_limits_cooldown() {
    let mut limits = DailyLimits::new(100, 100_000.0, 5);
    limits.last_trade_time = Some(Utc::now());
    assert!(limits.check(100.0).is_err());
}

#[test]
fn test_format_final_report() {
    let plan = ExecutionPlan::NoTrade {
        reason: "test".into(),
    };
    let state = ExecutionState::new(&plan);
    let report = format_final_report(&state);
    assert!(report.contains("Execution Report"));
    assert!(report.contains("Status"));
}

#[test]
fn test_daily_limits_record_trade() {
    let mut limits = DailyLimits::new(10, 10_000.0, 0);
    limits.record_trade(500.0);
    assert_eq!(limits.trades_today, 1);
    assert!((limits.usd_today - 500.0).abs() < 1e-6);
    assert!(limits.last_trade_time.is_some());
}

#[test]
fn test_bracket_price_calc_buy() {
    let entry: f64 = 100.0;
    let sl_pct: f64 = 1.5;
    let tp_pct: f64 = 3.0;
    let sl_price = entry * (1.0 - sl_pct / 100.0);
    let tp_price = entry * (1.0 + tp_pct / 100.0);
    assert!((sl_price - 98.5).abs() < 1e-6);
    assert!((tp_price - 103.0).abs() < 1e-6);
}

#[test]
fn test_bracket_price_calc_sell() {
    let entry: f64 = 100.0;
    let sl_pct: f64 = 1.5;
    let tp_pct: f64 = 3.0;
    let sl_price = entry * (1.0 + sl_pct / 100.0);
    let tp_price = entry * (1.0 - tp_pct / 100.0);
    assert!((sl_price - 101.5).abs() < 1e-6);
    assert!((tp_price - 97.0).abs() < 1e-6);
}

#[test]
fn test_max_positions_check() {
    assert!(check_max_positions(2, 3).is_ok());
    assert!(check_max_positions(3, 3).is_err());
    assert!(check_max_positions(5, 3).is_err());
}

#[test]
fn test_pnl_cutoff_check() {
    // -3% loss on $10k portfolio = -$300, cutoff at 5% -> OK
    assert!(check_daily_pnl_cutoff(-300.0, 10_000.0, 5.0).is_ok());
    // -6% loss on $10k portfolio = -$600, cutoff at 5% -> blocked
    assert!(check_daily_pnl_cutoff(-600.0, 10_000.0, 5.0).is_err());
    // Zero portfolio -> always OK (avoids division by zero)
    assert!(check_daily_pnl_cutoff(-1000.0, 0.0, 5.0).is_ok());
}
