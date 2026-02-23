//! Execution plan types — TWAP and Immediate order strategies.

use serde::{Deserialize, Serialize};

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
        }
    }
}

/// Execution plan produced by the planning engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionPlan {
    Immediate(ImmediatePlan),
    Twap(TwapPlan),
    NoTrade { reason: String },
}

/// Immediate (single market order) plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmediatePlan {
    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub estimated_price: f64,
    pub estimated_usd: f64,
}

/// TWAP (time-weighted average price) plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapPlan {
    pub symbol: String,
    pub side: Side,
    pub total_quantity: f64,
    pub slices: Vec<OrderSlice>,
    pub interval_secs: u64,
    pub estimated_price: f64,
    pub estimated_total_usd: f64,
}

/// A single slice in a TWAP plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSlice {
    pub index: u32,
    pub quantity: f64,
    pub status: SliceStatus,
}

/// Status of an individual TWAP slice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SliceStatus {
    Pending,
    Filled,
    PartialFill,
    Failed,
    Skipped,
}

/// Plan an execution strategy based on order size relative to daily volume.
///
/// - < 0.1% of daily volume → Immediate
/// - < 1% of daily volume → TWAP
/// - >= 1% of daily volume → NoTrade (too large)
pub fn plan_execution(
    symbol: &str,
    side: Side,
    quantity: f64,
    price: f64,
    daily_volume: f64,
) -> ExecutionPlan {
    let order_usd = quantity * price;
    let volume_usd = daily_volume * price;

    if volume_usd <= 0.0 {
        return ExecutionPlan::NoTrade {
            reason: "No volume data available".into(),
        };
    }

    let pct_of_volume = order_usd / volume_usd;

    if pct_of_volume < 0.001 {
        // < 0.1% of daily volume → Immediate
        ExecutionPlan::Immediate(ImmediatePlan {
            symbol: symbol.to_string(),
            side,
            quantity,
            estimated_price: price,
            estimated_usd: order_usd,
        })
    } else if pct_of_volume < 0.01 {
        // < 1% of daily volume → TWAP
        let num_slices = ((pct_of_volume * 1000.0).ceil() as u32).clamp(3, 20);
        let slice_qty = quantity / num_slices as f64;

        let slices = (0..num_slices)
            .map(|i| OrderSlice {
                index: i,
                quantity: slice_qty,
                status: SliceStatus::Pending,
            })
            .collect();

        ExecutionPlan::Twap(TwapPlan {
            symbol: symbol.to_string(),
            side,
            total_quantity: quantity,
            slices,
            interval_secs: 60,
            estimated_price: price,
            estimated_total_usd: order_usd,
        })
    } else {
        ExecutionPlan::NoTrade {
            reason: format!(
                "Order is {:.2}% of daily volume — too large for safe execution",
                pct_of_volume * 100.0
            ),
        }
    }
}

/// Format an execution plan as a human-readable summary.
pub fn format_plan(plan: &ExecutionPlan) -> String {
    match plan {
        ExecutionPlan::Immediate(p) => {
            format!(
                "IMMEDIATE: {} {:.6} {} @ ~${:.2} (est. ${:.2})",
                p.side, p.quantity, p.symbol, p.estimated_price, p.estimated_usd,
            )
        }
        ExecutionPlan::Twap(p) => {
            format!(
                "TWAP: {} {:.6} {} in {} slices over {}s @ ~${:.2} (est. ${:.2})",
                p.side,
                p.total_quantity,
                p.symbol,
                p.slices.len(),
                p.slices.len() as u64 * p.interval_secs,
                p.estimated_price,
                p.estimated_total_usd,
            )
        }
        ExecutionPlan::NoTrade { reason } => {
            format!("NO TRADE: {reason}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_order_immediate() {
        // 0.001 BTC at $50k with daily volume 1000 BTC = 0.0001% of volume
        let plan = plan_execution("BTCUSDT", Side::Buy, 0.001, 50_000.0, 1000.0);
        assert!(
            matches!(plan, ExecutionPlan::Immediate(_)),
            "Small order should be Immediate"
        );
    }

    #[test]
    fn test_medium_order_twap() {
        // 2 BTC at $50k with daily volume 1000 BTC = 0.2% of volume
        let plan = plan_execution("BTCUSDT", Side::Buy, 2.0, 50_000.0, 1000.0);
        assert!(
            matches!(plan, ExecutionPlan::Twap(_)),
            "Medium order should be TWAP"
        );
    }

    #[test]
    fn test_large_order_no_trade() {
        // 20 BTC at $50k with daily volume 1000 BTC = 2% of volume
        let plan = plan_execution("BTCUSDT", Side::Sell, 20.0, 50_000.0, 1000.0);
        assert!(
            matches!(plan, ExecutionPlan::NoTrade { .. }),
            "Large order should be NoTrade"
        );
    }

    #[test]
    fn test_twap_correct_slice_count() {
        let plan = plan_execution("BTCUSDT", Side::Buy, 5.0, 50_000.0, 1000.0);
        if let ExecutionPlan::Twap(twap) = plan {
            assert!(twap.slices.len() >= 3, "TWAP should have at least 3 slices");
            // All slices should be pending
            for slice in &twap.slices {
                assert_eq!(slice.status, SliceStatus::Pending);
            }
            // Total quantity should match
            let total: f64 = twap.slices.iter().map(|s| s.quantity).sum();
            assert!(
                (total - 5.0).abs() < 1e-6,
                "Slice quantities should sum to total: {total}"
            );
        } else {
            panic!("Expected TWAP plan");
        }
    }

    #[test]
    fn test_format_plan() {
        let plan = ExecutionPlan::Immediate(ImmediatePlan {
            symbol: "BTCUSDT".into(),
            side: Side::Buy,
            quantity: 0.01,
            estimated_price: 50_000.0,
            estimated_usd: 500.0,
        });
        let formatted = format_plan(&plan);
        assert!(formatted.contains("IMMEDIATE"));
        assert!(formatted.contains("BUY"));
        assert!(formatted.contains("BTCUSDT"));
    }

    #[test]
    fn test_no_volume_data() {
        let plan = plan_execution("BTCUSDT", Side::Buy, 1.0, 50_000.0, 0.0);
        assert!(matches!(plan, ExecutionPlan::NoTrade { .. }));
    }

    #[test]
    fn test_side_display() {
        assert_eq!(format!("{}", Side::Buy), "BUY");
        assert_eq!(format!("{}", Side::Sell), "SELL");
    }
}
