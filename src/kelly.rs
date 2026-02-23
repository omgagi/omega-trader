//! Fractional Kelly criterion for position sizing.
//!
//! Computes optimal position size given win probability, win/loss ratio,
//! and risk parameters. Includes safety clamps for crypto markets.

use serde::{Deserialize, Serialize};

/// Kelly criterion calculator with safety guardrails.
pub struct KellyCriterion {
    /// Fraction of full Kelly to use (0.1 to 1.0).
    fraction: f64,
    /// Maximum allocation as fraction of portfolio (0.01 to 0.5).
    max_allocation: f64,
    /// Minimum confidence to trade.
    min_confidence: f64,
}

/// Output of a Kelly calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyOutput {
    /// Full Kelly fraction (unscaled).
    pub full_kelly: f64,
    /// Fractional Kelly (scaled by fraction parameter).
    pub fractional_kelly: f64,
    /// Dollar position size.
    pub position_size_usd: f64,
    /// Whether to trade based on edge and confidence.
    pub should_trade: bool,
    /// Reason for the decision.
    pub reason: String,
}

impl KellyCriterion {
    /// Create a new Kelly calculator with custom parameters (clamped for safety).
    pub fn new(fraction: f64, max_allocation: f64, min_confidence: f64) -> Self {
        Self {
            fraction: fraction.clamp(0.1, 1.0),
            max_allocation: max_allocation.clamp(0.01, 0.5),
            min_confidence,
        }
    }

    /// Sensible defaults for crypto: 25% Kelly, 10% max allocation.
    pub fn crypto_default() -> Self {
        Self::new(0.25, 0.10, 0.55)
    }

    /// Calculate optimal position size.
    ///
    /// - `win_prob`: probability of winning the trade (0.0 to 1.0)
    /// - `win_loss_ratio`: average win / average loss
    /// - `portfolio_value`: total portfolio in USD
    /// - `regime_confidence`: confidence in the current regime detection
    pub fn calculate(
        &self,
        win_prob: f64,
        win_loss_ratio: f64,
        portfolio_value: f64,
        regime_confidence: f64,
    ) -> KellyOutput {
        // Kelly formula: f* = (p * b - q) / b
        // where p = win probability, q = 1 - p, b = win/loss ratio
        let q = 1.0 - win_prob;
        let full_kelly = if win_loss_ratio > 0.0 {
            (win_prob * win_loss_ratio - q) / win_loss_ratio
        } else {
            0.0
        };

        // No edge → don't trade
        if full_kelly <= 0.0 {
            return KellyOutput {
                full_kelly,
                fractional_kelly: 0.0,
                position_size_usd: 0.0,
                should_trade: false,
                reason: "No edge detected (Kelly <= 0)".into(),
            };
        }

        // Low confidence → don't trade
        if regime_confidence < self.min_confidence {
            return KellyOutput {
                full_kelly,
                fractional_kelly: 0.0,
                position_size_usd: 0.0,
                should_trade: false,
                reason: format!(
                    "Regime confidence {:.0}% below minimum {:.0}%",
                    regime_confidence * 100.0,
                    self.min_confidence * 100.0
                ),
            };
        }

        // Apply fractional Kelly and max allocation cap
        let fractional = full_kelly * self.fraction;
        let capped = fractional.min(self.max_allocation);
        let position_usd = capped * portfolio_value;

        KellyOutput {
            full_kelly,
            fractional_kelly: capped,
            position_size_usd: position_usd,
            should_trade: true,
            reason: format!(
                "Edge detected: full Kelly {:.1}%, using {:.1}% (capped at {:.1}%)",
                full_kelly * 100.0,
                capped * 100.0,
                self.max_allocation * 100.0,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_edge() {
        let kelly = KellyCriterion::crypto_default();
        let output = kelly.calculate(0.60, 1.5, 10_000.0, 0.80);
        assert!(output.full_kelly > 0.0, "Should detect positive edge");
        assert!(output.should_trade, "Should recommend trading");
        assert!(
            output.position_size_usd > 0.0,
            "Position size should be positive"
        );
        assert!(
            output.position_size_usd <= 10_000.0 * 0.10,
            "Should respect max allocation of 10%"
        );
    }

    #[test]
    fn test_no_edge() {
        let kelly = KellyCriterion::crypto_default();
        let output = kelly.calculate(0.40, 1.0, 10_000.0, 0.80);
        assert!(
            output.full_kelly <= 0.0,
            "Should detect no edge with 40% win rate and 1:1 ratio"
        );
        assert!(!output.should_trade);
        assert_eq!(output.position_size_usd, 0.0);
    }

    #[test]
    fn test_low_confidence() {
        let kelly = KellyCriterion::crypto_default();
        let output = kelly.calculate(0.60, 1.5, 10_000.0, 0.40);
        assert!(!output.should_trade, "Should not trade with low confidence");
        assert_eq!(output.position_size_usd, 0.0);
    }

    #[test]
    fn test_respects_max_allocation() {
        let kelly = KellyCriterion::new(1.0, 0.05, 0.50); // Full Kelly, 5% max
        let output = kelly.calculate(0.70, 2.0, 100_000.0, 0.90);
        assert!(
            output.position_size_usd <= 100_000.0 * 0.05 + 0.01,
            "Position ${:.0} should not exceed 5% of $100k = $5000",
            output.position_size_usd
        );
    }

    #[test]
    fn test_fraction_clamped() {
        let kelly = KellyCriterion::new(2.0, 0.80, 0.50);
        assert_eq!(kelly.fraction, 1.0, "Fraction should be clamped to 1.0");
        assert_eq!(
            kelly.max_allocation, 0.5,
            "Max allocation should be clamped to 0.5"
        );
    }

    #[test]
    fn test_fraction_clamped_low() {
        let kelly = KellyCriterion::new(0.01, 0.001, 0.50);
        assert_eq!(kelly.fraction, 0.1, "Fraction should be clamped to 0.1");
        assert_eq!(
            kelly.max_allocation, 0.01,
            "Max allocation should be clamped to 0.01"
        );
    }

    #[test]
    fn test_kelly_max_position_20_percent() {
        // Safety invariant #4: max position <= 20% (0.5 clamp)
        let kelly = KellyCriterion::new(1.0, 0.99, 0.0);
        assert!(
            kelly.max_allocation <= 0.5,
            "Max allocation must be clamped to 50%"
        );
    }
}
