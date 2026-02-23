//! Broker abstraction — trait + shared types + factory.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[cfg(feature = "ibkr")]
pub mod ibkr;

// Re-export IbkrBroker when feature is enabled.
#[cfg(feature = "ibkr")]
pub use ibkr::IbkrBroker;

use crate::execution::Side;

/// Asset class for multi-instrument support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetClass {
    Stock,
    Forex,
    Crypto,
}

impl std::str::FromStr for AssetClass {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "stock" | "stk" => Ok(Self::Stock),
            "forex" | "fx" | "cash" => Ok(Self::Forex),
            "crypto" => Ok(Self::Crypto),
            _ => anyhow::bail!("Unknown asset class '{s}'. Use: stock, forex/fx, crypto"),
        }
    }
}

impl std::fmt::Display for AssetClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stock => write!(f, "stock"),
            Self::Forex => write!(f, "forex"),
            Self::Crypto => write!(f, "crypto"),
        }
    }
}

/// Fill result from a broker order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFill {
    pub order_id: i32,
    pub filled_qty: f64,
    pub filled_usd: f64,
}

/// Position information from a broker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionInfo {
    /// Account ID.
    pub account: String,
    /// Instrument symbol.
    pub symbol: String,
    /// Security type (e.g. "STK", "CRYPTO", "CASH").
    pub security_type: String,
    /// Position quantity (positive = long, negative = short).
    pub quantity: f64,
    /// Average cost per unit.
    pub avg_cost: f64,
}

/// Daily P&L for an account.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyPnl {
    /// Daily profit/loss.
    pub daily_pnl: f64,
    /// Unrealized P&L (open positions).
    pub unrealized_pnl: Option<f64>,
    /// Realized P&L (closed positions).
    pub realized_pnl: Option<f64>,
}

/// Broker-agnostic trait for order execution and market data.
#[async_trait]
pub trait Broker: Send + Sync {
    /// Downcast support for broker-specific methods.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Check if the broker connection is reachable.
    async fn check_connection(&self) -> bool;

    /// Get current price for a symbol.
    async fn get_price(&self, symbol: &str, asset_class: AssetClass) -> Result<f64>;

    /// Place a market order. Returns fill information.
    async fn place_order(
        &self,
        symbol: &str,
        asset_class: AssetClass,
        side: Side,
        qty: f64,
    ) -> Result<OrderFill>;

    /// Get all open positions.
    async fn get_positions(&self) -> Result<Vec<PositionInfo>>;

    /// Get daily P&L for an account.
    async fn get_daily_pnl(&self, account: &str) -> Result<DailyPnl>;
}

/// Build a broker by name.
pub fn build_broker(name: &str, host: &str, port: u16, client_id: i32) -> Result<Box<dyn Broker>> {
    match name {
        #[cfg(feature = "ibkr")]
        "ibkr" => Ok(Box::new(IbkrBroker::new(host, port, client_id))),
        _ => anyhow::bail!("Unknown broker '{name}'. Available: ibkr"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_class_parse() {
        assert_eq!("stock".parse::<AssetClass>().unwrap(), AssetClass::Stock);
        assert_eq!("stk".parse::<AssetClass>().unwrap(), AssetClass::Stock);
        assert_eq!("forex".parse::<AssetClass>().unwrap(), AssetClass::Forex);
        assert_eq!("fx".parse::<AssetClass>().unwrap(), AssetClass::Forex);
        assert_eq!("cash".parse::<AssetClass>().unwrap(), AssetClass::Forex);
        assert_eq!("crypto".parse::<AssetClass>().unwrap(), AssetClass::Crypto);
        assert!("invalid".parse::<AssetClass>().is_err());
    }

    #[test]
    fn test_asset_class_display() {
        assert_eq!(AssetClass::Stock.to_string(), "stock");
        assert_eq!(AssetClass::Forex.to_string(), "forex");
        assert_eq!(AssetClass::Crypto.to_string(), "crypto");
    }

    #[test]
    fn test_position_info_serde() {
        let pos = PositionInfo {
            account: "DU1234567".into(),
            symbol: "AAPL".into(),
            security_type: "STK".into(),
            quantity: 100.0,
            avg_cost: 150.50,
        };
        let json = serde_json::to_string(&pos).unwrap();
        let recovered: PositionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.account, "DU1234567");
        assert_eq!(recovered.symbol, "AAPL");
        assert!((recovered.quantity - 100.0).abs() < f64::EPSILON);
        assert!((recovered.avg_cost - 150.50).abs() < f64::EPSILON);
    }

    #[test]
    fn test_daily_pnl_serde() {
        let pnl = DailyPnl {
            daily_pnl: -250.50,
            unrealized_pnl: Some(-100.0),
            realized_pnl: Some(-150.50),
        };
        let json = serde_json::to_string(&pnl).unwrap();
        let recovered: DailyPnl = serde_json::from_str(&json).unwrap();
        assert!((recovered.daily_pnl - (-250.50)).abs() < f64::EPSILON);
        assert_eq!(recovered.unrealized_pnl, Some(-100.0));
        assert_eq!(recovered.realized_pnl, Some(-150.50));
    }

    #[test]
    fn test_order_fill_serde() {
        let fill = OrderFill {
            order_id: 42,
            filled_qty: 10.5,
            filled_usd: 1575.0,
        };
        let json = serde_json::to_string(&fill).unwrap();
        let recovered: OrderFill = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.order_id, 42);
        assert!((recovered.filled_qty - 10.5).abs() < f64::EPSILON);
    }

    #[cfg(feature = "ibkr")]
    #[test]
    fn test_build_broker_ibkr() {
        let broker = build_broker("ibkr", "127.0.0.1", 7497, 1);
        assert!(broker.is_ok());
    }

    #[test]
    fn test_build_broker_unknown() {
        let broker = build_broker("unknown", "127.0.0.1", 7497, 1);
        assert!(broker.is_err());
    }
}
