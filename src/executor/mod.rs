//! Live order executor with circuit breaker, daily limits, and crash recovery.
//!
//! Uses the `Broker` trait for order placement — broker-agnostic.

mod guardrails;

#[cfg(test)]
mod tests;

pub use guardrails::*;

use crate::broker::Broker;
use crate::execution::{ExecutionPlan, ImmediatePlan, TwapPlan};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Max price deviation from entry before aborting (default: 2%).
    pub max_deviation_pct: f64,
    /// Max consecutive slice failures before aborting (default: 3).
    pub max_consecutive_failures: u32,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            max_deviation_pct: 0.02,
            max_consecutive_failures: 3,
        }
    }
}

/// Safety limits for daily trading.
#[derive(Debug, Clone)]
pub struct DailyLimits {
    /// Maximum number of trades per day.
    pub max_trades: u32,
    /// Maximum total USD per day.
    pub max_usd: f64,
    /// Minimum cooldown between trades in minutes.
    pub cooldown_minutes: u32,
    /// Trades executed today.
    pub trades_today: u32,
    /// USD traded today.
    pub usd_today: f64,
    /// Last trade timestamp.
    pub last_trade_time: Option<DateTime<Utc>>,
}

impl DailyLimits {
    /// Create new limits from config values.
    pub fn new(max_trades: u32, max_usd: f64, cooldown_minutes: u32) -> Self {
        Self {
            max_trades,
            max_usd,
            cooldown_minutes,
            trades_today: 0,
            usd_today: 0.0,
            last_trade_time: None,
        }
    }

    /// Check if a trade is allowed. Returns `Err` with reason if blocked.
    pub fn check(&self, trade_usd: f64) -> Result<()> {
        if self.trades_today >= self.max_trades {
            anyhow::bail!(
                "Daily trade limit reached ({}/{})",
                self.trades_today,
                self.max_trades
            );
        }

        if self.usd_today + trade_usd > self.max_usd {
            anyhow::bail!(
                "Daily USD limit would be exceeded (${:.0} + ${:.0} > ${:.0})",
                self.usd_today,
                trade_usd,
                self.max_usd
            );
        }

        if let Some(last) = self.last_trade_time {
            let elapsed = Utc::now() - last;
            let cooldown = chrono::Duration::minutes(self.cooldown_minutes as i64);
            if elapsed < cooldown {
                let remaining = cooldown - elapsed;
                anyhow::bail!("Cooldown active: {}s remaining", remaining.num_seconds());
            }
        }

        Ok(())
    }

    /// Record a completed trade.
    pub fn record_trade(&mut self, usd: f64) {
        self.trades_today += 1;
        self.usd_today += usd;
        self.last_trade_time = Some(Utc::now());
    }
}

/// Execution status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    Confirmed,
    Running,
    Completed,
    PartialFill,
    Aborted,
    Failed,
}

/// Persistent execution state for crash recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    pub plan_json: String,
    pub slices_completed: u32,
    pub total_slices: u32,
    pub total_filled_qty: f64,
    pub total_filled_usd: f64,
    pub status: ExecutionStatus,
    pub order_ids: Vec<i64>,
    pub errors: Vec<String>,
    pub abort_reason: Option<String>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ExecutionState {
    /// Create initial state for a plan.
    pub fn new(plan: &ExecutionPlan) -> Self {
        let total_slices = match plan {
            ExecutionPlan::Immediate(_) => 1,
            ExecutionPlan::Twap(t) => t.slices.len() as u32,
            ExecutionPlan::NoTrade { .. } => 0,
        };
        let plan_json = serde_json::to_string(plan).unwrap_or_default();

        Self {
            plan_json,
            slices_completed: 0,
            total_slices,
            total_filled_qty: 0.0,
            total_filled_usd: 0.0,
            status: ExecutionStatus::Pending,
            order_ids: Vec::new(),
            errors: Vec::new(),
            abort_reason: None,
            started_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

/// Live executor with safety guardrails, using the `Broker` trait.
pub struct Executor {
    broker: Box<dyn Broker>,
    circuit_breaker: CircuitBreaker,
    daily_limits: DailyLimits,
}

impl Executor {
    /// Create a new executor with a broker.
    pub fn new(
        broker: Box<dyn Broker>,
        circuit_breaker: CircuitBreaker,
        daily_limits: DailyLimits,
    ) -> Self {
        Self {
            broker,
            circuit_breaker,
            daily_limits,
        }
    }

    /// Execute a plan. Returns the final execution state.
    pub async fn execute(&mut self, plan: &ExecutionPlan) -> ExecutionState {
        let mut state = ExecutionState::new(plan);

        match plan {
            ExecutionPlan::NoTrade { reason } => {
                state.status = ExecutionStatus::Completed;
                state.abort_reason = Some(reason.clone());
                return state;
            }
            ExecutionPlan::Immediate(p) => {
                if let Err(e) = self.daily_limits.check(p.estimated_usd) {
                    state.status = ExecutionStatus::Aborted;
                    state.abort_reason = Some(e.to_string());
                    return state;
                }
                state.status = ExecutionStatus::Running;
                self.execute_immediate(p, &mut state).await;
            }
            ExecutionPlan::Twap(p) => {
                if let Err(e) = self.daily_limits.check(p.estimated_total_usd) {
                    state.status = ExecutionStatus::Aborted;
                    state.abort_reason = Some(e.to_string());
                    return state;
                }
                state.status = ExecutionStatus::Running;
                self.execute_twap(p, &mut state).await;
            }
        }

        state.updated_at = Utc::now();
        state
    }

    /// Execute a single immediate order via the broker.
    async fn execute_immediate(&mut self, plan: &ImmediatePlan, state: &mut ExecutionState) {
        match self
            .broker
            .place_order(&plan.symbol, plan.asset_class, plan.side, plan.quantity)
            .await
        {
            Ok(fill) => {
                state.order_ids.push(fill.order_id as i64);
                state.total_filled_qty += fill.filled_qty;
                state.total_filled_usd += fill.filled_usd;
                state.slices_completed = 1;
                state.status = ExecutionStatus::Completed;
                self.daily_limits.record_trade(fill.filled_usd);
                info!(
                    "quant: immediate order filled: {:.6} {} (${:.2})",
                    fill.filled_qty, plan.symbol, fill.filled_usd
                );
            }
            Err(e) => {
                state.status = ExecutionStatus::Failed;
                state.errors.push(e.to_string());
                error!("quant: immediate order failed: {e}");
            }
        }
    }

    /// Execute a TWAP plan slice by slice via the broker.
    async fn execute_twap(&mut self, plan: &TwapPlan, state: &mut ExecutionState) {
        let entry_price = plan.estimated_price;
        let mut consecutive_failures: u32 = 0;

        for (i, slice) in plan.slices.iter().enumerate() {
            // Circuit breaker: check price deviation.
            match self.broker.get_price(&plan.symbol, plan.asset_class).await {
                Ok(current_price) => {
                    if entry_price > 0.0 {
                        let deviation = (current_price - entry_price).abs() / entry_price;
                        if deviation > self.circuit_breaker.max_deviation_pct {
                            state.status = ExecutionStatus::Aborted;
                            state.abort_reason = Some(format!(
                                "Circuit breaker: price deviated {:.2}% (max {:.2}%)",
                                deviation * 100.0,
                                self.circuit_breaker.max_deviation_pct * 100.0
                            ));
                            warn!("quant: {}", state.abort_reason.as_ref().unwrap());
                            return;
                        }
                    }
                }
                Err(e) => {
                    warn!("quant: failed to check price for circuit breaker: {e}");
                }
            }

            // Execute slice.
            match self
                .broker
                .place_order(&plan.symbol, plan.asset_class, plan.side, slice.quantity)
                .await
            {
                Ok(fill) => {
                    state.order_ids.push(fill.order_id as i64);
                    state.total_filled_qty += fill.filled_qty;
                    state.total_filled_usd += fill.filled_usd;
                    state.slices_completed += 1;
                    consecutive_failures = 0;
                    info!(
                        "quant: TWAP slice {}/{}: {:.6} filled (${:.2})",
                        i + 1,
                        plan.slices.len(),
                        fill.filled_qty,
                        fill.filled_usd
                    );
                }
                Err(e) => {
                    consecutive_failures += 1;
                    state.errors.push(format!("Slice {}: {e}", i + 1));
                    error!("quant: TWAP slice {} failed: {e}", i + 1);

                    if consecutive_failures >= self.circuit_breaker.max_consecutive_failures {
                        state.status = ExecutionStatus::Aborted;
                        state.abort_reason = Some(format!(
                            "Circuit breaker: {consecutive_failures} consecutive failures"
                        ));
                        warn!("quant: {}", state.abort_reason.as_ref().unwrap());
                        return;
                    }
                }
            }

            // Progress update every 5 slices.
            if (i + 1) % 5 == 0 {
                info!(
                    "quant: TWAP progress: {}/{} slices, {:.6} filled",
                    state.slices_completed,
                    plan.slices.len(),
                    state.total_filled_qty,
                );
            }

            // Wait between slices (except last).
            if i + 1 < plan.slices.len() {
                tokio::time::sleep(std::time::Duration::from_secs(plan.interval_secs)).await;
            }
        }

        // Determine final status.
        if state.slices_completed == plan.slices.len() as u32 {
            state.status = ExecutionStatus::Completed;
            self.daily_limits.record_trade(state.total_filled_usd);
        } else if state.slices_completed > 0 {
            state.status = ExecutionStatus::PartialFill;
            self.daily_limits.record_trade(state.total_filled_usd);
        } else {
            state.status = ExecutionStatus::Failed;
        }
    }
}
