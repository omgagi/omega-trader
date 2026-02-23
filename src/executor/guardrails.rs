//! Safety guardrails, state persistence, and execution reporting.

use super::ExecutionState;
use anyhow::Result;

/// Check if adding a position would exceed the maximum position count.
pub fn check_max_positions(current: usize, max: usize) -> Result<()> {
    if current >= max {
        anyhow::bail!(
            "Max positions limit reached ({current}/{max}). Close a position before opening new ones."
        );
    }
    Ok(())
}

/// Check if daily P&L has breached the cutoff threshold.
pub fn check_daily_pnl_cutoff(daily_pnl: f64, portfolio: f64, cutoff_pct: f64) -> Result<()> {
    if portfolio > 0.0 {
        let pnl_pct = (daily_pnl / portfolio) * 100.0;
        if pnl_pct <= -cutoff_pct.abs() {
            anyhow::bail!(
                "Daily P&L cutoff breached: {pnl_pct:.2}% (limit: -{:.1}%). Trading halted.",
                cutoff_pct.abs()
            );
        }
    }
    Ok(())
}

/// Serialize execution state to JSON for crash recovery.
pub fn persist_state(state: &ExecutionState) -> Result<String> {
    Ok(serde_json::to_string_pretty(state)?)
}

/// Recover execution state from JSON.
pub fn recover_state(json: &str) -> Result<ExecutionState> {
    Ok(serde_json::from_str(json)?)
}

/// Format a final execution report.
pub fn format_final_report(state: &ExecutionState) -> String {
    format!(
        "Execution Report\n\
         \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n\
         Status: {:?}\n\
         Slices: {}/{}\n\
         Filled: {:.6} (${:.2})\n\
         Orders: {}\n\
         Errors: {}\n\
         {}Started: {}\n\
         Duration: {}s",
        state.status,
        state.slices_completed,
        state.total_slices,
        state.total_filled_qty,
        state.total_filled_usd,
        state.order_ids.len(),
        state.errors.len(),
        state
            .abort_reason
            .as_ref()
            .map(|r| format!("Abort reason: {r}\n"))
            .unwrap_or_default(),
        state.started_at.format("%H:%M:%S"),
        (state.updated_at - state.started_at).num_seconds(),
    )
}
