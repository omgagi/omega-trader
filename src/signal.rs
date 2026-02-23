//! Quantitative signal output types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Market regime detected by the HMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Regime {
    Bull,
    Bear,
    Lateral,
}

/// Trade direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
    Hold,
}

/// Recommended action with urgency/parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Long { urgency: f64 },
    Short { urgency: f64 },
    Hold,
    ReducePosition { by_percent: f64 },
    Exit,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Long { urgency } => write!(f, "LONG (urgency: {:.0}%)", urgency * 100.0),
            Self::Short { urgency } => write!(f, "SHORT (urgency: {:.0}%)", urgency * 100.0),
            Self::Hold => write!(f, "HOLD"),
            Self::ReducePosition { by_percent } => {
                write!(f, "REDUCE {by_percent:.0}%")
            }
            Self::Exit => write!(f, "EXIT"),
        }
    }
}

/// Execution strategy recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    Twap { slices: u32, interval_secs: u64 },
    Immediate,
    DontTrade,
}

/// Regime probability distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeProbabilities {
    pub bull: f64,
    pub bear: f64,
    pub lateral: f64,
}

/// Hurst exponent interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HurstInterpretation {
    MeanReverting,
    Random,
    Trending,
}

impl fmt::Display for HurstInterpretation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeanReverting => write!(f, "Mean-Reverting"),
            Self::Random => write!(f, "Random Walk"),
            Self::Trending => write!(f, "Trending"),
        }
    }
}

/// Complete quantitative signal output from the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantSignal {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub raw_price: f64,
    pub filtered_price: f64,
    pub trend: f64,
    pub regime: Regime,
    pub regime_probabilities: RegimeProbabilities,
    pub hurst_exponent: f64,
    pub hurst_interpretation: HurstInterpretation,
    pub merton_allocation: f64,
    pub kelly_fraction: f64,
    pub kelly_position_usd: f64,
    pub kelly_should_trade: bool,
    pub direction: Direction,
    pub action: Action,
    pub execution: ExecutionStrategy,
    pub confidence: f64,
    pub reasoning: String,
}

impl QuantSignal {
    /// Returns `true` when the signal represents a critical event that should
    /// break through even when the user is not in a trading context — e.g.
    /// an EXIT action or very high-urgency entry/reduce (≥ 80 %).
    pub fn is_critical(&self) -> bool {
        match &self.action {
            Action::Exit => true,
            Action::ReducePosition { by_percent } => *by_percent >= 50.0,
            Action::Long { urgency } | Action::Short { urgency } => *urgency >= 0.80,
            Action::Hold => false,
        }
    }
}

impl fmt::Display for QuantSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ${:.2} | {:?} | {:?} | {} | conf: {:.0}%",
            self.timestamp.format("%H:%M:%S"),
            self.symbol,
            self.filtered_price,
            self.regime,
            self.direction,
            self.action,
            self.confidence * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_roundtrip() {
        let signal = QuantSignal {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".into(),
            raw_price: 50_000.0,
            filtered_price: 49_980.0,
            trend: 0.001,
            regime: Regime::Bull,
            regime_probabilities: RegimeProbabilities {
                bull: 0.7,
                bear: 0.1,
                lateral: 0.2,
            },
            hurst_exponent: 0.5,
            hurst_interpretation: HurstInterpretation::Random,
            merton_allocation: 0.65,
            kelly_fraction: 0.08,
            kelly_position_usd: 800.0,
            kelly_should_trade: true,
            direction: Direction::Long,
            action: Action::Long { urgency: 0.7 },
            execution: ExecutionStrategy::Immediate,
            confidence: 0.56,
            reasoning: "Bull trend detected".into(),
        };

        let json = serde_json::to_string(&signal).unwrap();
        let deserialized: QuantSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.symbol, "BTCUSDT");
        assert_eq!(deserialized.raw_price, 50_000.0);
        assert!(matches!(deserialized.regime, Regime::Bull));
        assert!(matches!(deserialized.direction, Direction::Long));
    }

    #[test]
    fn test_display_formatting() {
        let signal = QuantSignal {
            timestamp: Utc::now(),
            symbol: "ETHUSDT".into(),
            raw_price: 3000.0,
            filtered_price: 2998.0,
            trend: -0.001,
            regime: Regime::Bear,
            regime_probabilities: RegimeProbabilities {
                bull: 0.1,
                bear: 0.8,
                lateral: 0.1,
            },
            hurst_exponent: 0.5,
            hurst_interpretation: HurstInterpretation::Random,
            merton_allocation: -0.3,
            kelly_fraction: 0.0,
            kelly_position_usd: 0.0,
            kelly_should_trade: false,
            direction: Direction::Short,
            action: Action::Hold,
            execution: ExecutionStrategy::DontTrade,
            confidence: 0.4,
            reasoning: "Bear regime but Kelly says no".into(),
        };

        let display = format!("{signal}");
        assert!(display.contains("ETHUSDT"));
        assert!(display.contains("Bear"));
        assert!(display.contains("HOLD"));
    }

    #[test]
    fn test_action_display() {
        assert_eq!(
            format!("{}", Action::Long { urgency: 0.85 }),
            "LONG (urgency: 85%)"
        );
        assert_eq!(
            format!("{}", Action::Short { urgency: 0.6 }),
            "SHORT (urgency: 60%)"
        );
        assert_eq!(format!("{}", Action::Hold), "HOLD");
        assert_eq!(
            format!("{}", Action::ReducePosition { by_percent: 25.0 }),
            "REDUCE 25%"
        );
        assert_eq!(format!("{}", Action::Exit), "EXIT");
    }

    #[test]
    fn test_is_critical() {
        // EXIT is always critical.
        let mut sig = make_signal(Action::Exit);
        assert!(sig.is_critical());

        // High-urgency Long is critical.
        sig.action = Action::Long { urgency: 0.85 };
        assert!(sig.is_critical());

        // Low-urgency Long is not.
        sig.action = Action::Long { urgency: 0.5 };
        assert!(!sig.is_critical());

        // High-urgency Short is critical.
        sig.action = Action::Short { urgency: 0.80 };
        assert!(sig.is_critical());

        // ReducePosition >= 50% is critical.
        sig.action = Action::ReducePosition { by_percent: 60.0 };
        assert!(sig.is_critical());

        // ReducePosition < 50% is not.
        sig.action = Action::ReducePosition { by_percent: 30.0 };
        assert!(!sig.is_critical());

        // Hold is never critical.
        sig.action = Action::Hold;
        assert!(!sig.is_critical());
    }

    /// Helper to build a test signal with a given action.
    fn make_signal(action: Action) -> QuantSignal {
        QuantSignal {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".into(),
            raw_price: 67_000.0,
            filtered_price: 66_990.0,
            trend: -0.001,
            regime: Regime::Bear,
            regime_probabilities: RegimeProbabilities {
                bull: 0.2,
                bear: 0.6,
                lateral: 0.2,
            },
            hurst_exponent: 0.48,
            hurst_interpretation: HurstInterpretation::Trending,
            merton_allocation: -0.15,
            kelly_fraction: 0.02,
            kelly_position_usd: 200.0,
            kelly_should_trade: true,
            direction: Direction::Short,
            action,
            execution: ExecutionStrategy::Immediate,
            confidence: 0.61,
            reasoning: "Test signal".into(),
        }
    }

    #[test]
    fn test_hurst_interpretation_display() {
        assert_eq!(
            format!("{}", HurstInterpretation::MeanReverting),
            "Mean-Reverting"
        );
        assert_eq!(format!("{}", HurstInterpretation::Random), "Random Walk");
        assert_eq!(format!("{}", HurstInterpretation::Trending), "Trending");
    }
}
