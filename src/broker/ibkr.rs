//! IBKR broker implementation — market data, order execution, positions, P&L.
//!
//! Implements the `Broker` trait and provides IBKR-specific methods for
//! scanner, price feed, and contract building.

use super::{AssetClass, Broker, DailyPnl, OrderFill, PositionInfo};
use crate::execution::Side;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{error, info, warn};

/// IBKR connection configuration.
#[derive(Debug, Clone)]
pub struct IbkrConfig {
    /// TWS/Gateway host (default: "127.0.0.1").
    pub host: String,
    /// TWS port (paper: 7497, live: 7496). IB Gateway: paper 4002, live 4001.
    pub port: u16,
    /// Unique client ID per connection.
    pub client_id: i32,
}

impl IbkrConfig {
    /// Paper trading configuration (TWS port 7497).
    pub fn paper() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 7497,
            client_id: 1,
        }
    }

    /// Live trading configuration (port 4001).
    pub fn live() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 4001,
            client_id: 1,
        }
    }

    /// Connection URL in `host:port` format.
    pub fn connection_url(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// IBKR broker implementing the `Broker` trait.
pub struct IbkrBroker {
    /// Connection configuration.
    pub config: IbkrConfig,
}

impl IbkrBroker {
    /// Create a new IBKR broker.
    pub fn new(host: &str, port: u16, client_id: i32) -> Self {
        Self {
            config: IbkrConfig {
                host: host.to_string(),
                port,
                client_id,
            },
        }
    }

    /// Build an IBKR contract for the given symbol and asset class.
    ///
    /// - Stock: symbol = `"AAPL"`
    /// - Forex: symbol = `"EUR/USD"` (split on `/`)
    /// - Crypto: symbol = `"BTC"`
    pub fn build_contract(
        symbol: &str,
        asset_class: AssetClass,
    ) -> Result<ibapi::contracts::Contract> {
        use ibapi::contracts::Contract;

        match asset_class {
            AssetClass::Stock => Ok(Contract::stock(symbol).build()),
            AssetClass::Forex => {
                let parts: Vec<&str> = symbol.split('/').collect();
                if parts.len() != 2 {
                    anyhow::bail!(
                        "Forex symbol must be in BASE/QUOTE format (e.g. EUR/USD), got: {symbol}"
                    );
                }
                Ok(Contract::forex(parts[0], parts[1]).build())
            }
            AssetClass::Crypto => Ok(Contract::crypto(symbol).build()),
        }
    }

    /// Start a real-time price feed for a symbol via TWS API.
    ///
    /// Returns a broadcast receiver that emits `PriceTick` events. The feed runs in
    /// a background task and reconnects automatically on disconnection.
    pub fn start_price_feed(
        &self,
        symbol: &str,
        asset_class: AssetClass,
    ) -> broadcast::Receiver<PriceTick> {
        let (tx, rx) = broadcast::channel(256);
        let symbol = symbol.to_string();
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                info!(
                    "quant: connecting to IB Gateway at {}",
                    config.connection_url()
                );

                match connect_and_stream(&symbol, &config, asset_class, &tx).await {
                    Ok(()) => {
                        info!("quant: IBKR feed ended normally for {symbol}");
                    }
                    Err(e) => {
                        error!("quant: IBKR connection failed: {e}");
                    }
                }

                if tx.receiver_count() == 0 {
                    info!("quant: all receivers dropped, stopping feed");
                    return;
                }

                warn!("quant: IBKR disconnected, reconnecting in 5s...");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        });

        rx
    }

    /// Run an IBKR market scanner to find instruments by criteria.
    pub async fn run_scanner(
        &self,
        scan_code: &str,
        instrument: &str,
        location: &str,
        count: i32,
        min_price: Option<f64>,
        min_volume: Option<i32>,
    ) -> Result<Vec<ScanResult>> {
        use ibapi::Client;
        use ibapi::scanner::ScannerSubscription;

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 300)
            .await
            .context("failed to connect to IB Gateway for scanner")?;

        let params = ScannerSubscription {
            number_of_rows: count,
            instrument: Some(instrument.to_string()),
            location_code: Some(location.to_string()),
            scan_code: Some(scan_code.to_string()),
            above_price: min_price,
            above_volume: min_volume,
            ..ScannerSubscription::default()
        };

        let filter = Vec::new();
        let mut subscription = client
            .scanner_subscription(&params, &filter)
            .await
            .context("failed to start scanner subscription")?;

        let mut results = Vec::new();
        let timeout_dur = std::time::Duration::from_secs(10);

        // Scanner yields Vec<ScannerData> batches — read until timeout.
        while let Ok(Some(Ok(batch))) = tokio::time::timeout(timeout_dur, subscription.next()).await
        {
            for data in &batch {
                let contract = &data.contract_details.contract;
                results.push(ScanResult {
                    rank: data.rank,
                    symbol: contract.symbol.to_string(),
                    security_type: contract.security_type.to_string(),
                    exchange: contract.exchange.to_string(),
                    currency: contract.currency.to_string(),
                });
            }
        }

        Ok(results)
    }

    /// Place a bracket order (market entry + take profit + stop loss).
    ///
    /// Creates 3 linked orders: parent MKT -> TP LMT -> SL STP.
    /// The stop loss order has `transmit=true` which triggers all three.
    pub async fn place_bracket_order(
        &self,
        contract: &ibapi::contracts::Contract,
        side: Side,
        quantity: f64,
        take_profit_price: f64,
        stop_loss_price: f64,
    ) -> Result<crate::executor::ExecutionState> {
        use crate::executor::{ExecutionState, ExecutionStatus};
        use ibapi::Client;
        use ibapi::orders::{Action, PlaceOrder, order_builder};

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 100)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed: {e}"))?;

        let parent_id = client
            .next_valid_order_id()
            .await
            .map_err(|e| anyhow::anyhow!("failed to get order ID: {e}"))?;

        let action = match side {
            Side::Buy => Action::Buy,
            Side::Sell => Action::Sell,
        };
        let reverse_action = match side {
            Side::Buy => Action::Sell,
            Side::Sell => Action::Buy,
        };

        // Parent: market order (transmit=false).
        let mut parent = order_builder::market_order(action, quantity);
        parent.order_id = parent_id;
        parent.transmit = false;

        // Take profit: limit order in opposite direction (transmit=false).
        let mut take_profit =
            order_builder::limit_order(reverse_action, quantity, take_profit_price);
        take_profit.order_id = parent_id + 1;
        take_profit.parent_id = parent_id;
        take_profit.transmit = false;

        // Stop loss: stop order in opposite direction (transmit=true triggers all).
        let mut stop_loss = order_builder::stop(reverse_action, quantity, stop_loss_price);
        stop_loss.order_id = parent_id + 2;
        stop_loss.parent_id = parent_id;
        stop_loss.transmit = true;

        let mut state = ExecutionState {
            plan_json: String::new(),
            slices_completed: 0,
            total_slices: 3,
            total_filled_qty: 0.0,
            total_filled_usd: 0.0,
            status: ExecutionStatus::Running,
            order_ids: vec![
                parent_id as i64,
                (parent_id + 1) as i64,
                (parent_id + 2) as i64,
            ],
            errors: Vec::new(),
            abort_reason: None,
            started_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        // Place parent order.
        let mut parent_notifications = client
            .place_order(parent_id, contract, &parent)
            .await
            .map_err(|e| anyhow::anyhow!("Bracket parent order failed: {e}"))?;

        // Place take profit.
        let _tp = client
            .place_order(parent_id + 1, contract, &take_profit)
            .await
            .map_err(|e| anyhow::anyhow!("Bracket take-profit order failed: {e}"))?;

        // Place stop loss (transmit=true triggers all orders).
        let _sl = client
            .place_order(parent_id + 2, contract, &stop_loss)
            .await
            .map_err(|e| anyhow::anyhow!("Bracket stop-loss order failed: {e}"))?;

        // Read parent order fill.
        while let Some(result) = parent_notifications.next().await {
            match result {
                Ok(PlaceOrder::ExecutionData(exec)) => {
                    state.total_filled_qty = exec.execution.cumulative_quantity;
                    let avg_price = exec.execution.average_price;
                    state.total_filled_usd = state.total_filled_qty * avg_price;
                }
                Ok(PlaceOrder::CommissionReport(_)) => break,
                Ok(_) => {}
                Err(e) => {
                    state.errors.push(e.to_string());
                    break;
                }
            }
        }

        state.slices_completed = 3;
        state.status = ExecutionStatus::Completed;
        state.updated_at = chrono::Utc::now();

        info!(
            "quant: bracket order placed: parent={}, TP={}, SL={}, filled={:.6}",
            parent_id,
            parent_id + 1,
            parent_id + 2,
            state.total_filled_qty
        );

        Ok(state)
    }

    /// Close a position with a market order.
    pub async fn close_position(
        &self,
        contract: &ibapi::contracts::Contract,
        quantity: f64,
        side: Side,
    ) -> Result<crate::executor::ExecutionState> {
        use crate::executor::{ExecutionState, ExecutionStatus};
        use ibapi::Client;
        use ibapi::orders::{Action, PlaceOrder, order_builder};

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 600)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed: {e}"))?;

        let action = match side {
            Side::Buy => Action::Buy,
            Side::Sell => Action::Sell,
        };
        let order = order_builder::market_order(action, quantity);

        let order_id = client
            .next_valid_order_id()
            .await
            .map_err(|e| anyhow::anyhow!("failed to get order ID: {e}"))?;

        let mut notifications = client
            .place_order(order_id, contract, &order)
            .await
            .map_err(|e| anyhow::anyhow!("Close order failed: {e}"))?;

        let mut state = ExecutionState {
            plan_json: String::new(),
            slices_completed: 0,
            total_slices: 1,
            total_filled_qty: 0.0,
            total_filled_usd: 0.0,
            status: ExecutionStatus::Running,
            order_ids: vec![order_id as i64],
            errors: Vec::new(),
            abort_reason: None,
            started_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        while let Some(result) = notifications.next().await {
            match result {
                Ok(PlaceOrder::ExecutionData(exec)) => {
                    state.total_filled_qty = exec.execution.cumulative_quantity;
                    let avg_price = exec.execution.average_price;
                    state.total_filled_usd = state.total_filled_qty * avg_price;
                }
                Ok(PlaceOrder::CommissionReport(_)) => break,
                Ok(_) => {}
                Err(e) => {
                    state.errors.push(e.to_string());
                    break;
                }
            }
        }

        state.slices_completed = 1;
        state.status = ExecutionStatus::Completed;
        state.updated_at = chrono::Utc::now();

        info!(
            "quant: position closed: {side:?} {:.6} (${:.2})",
            state.total_filled_qty, state.total_filled_usd
        );

        Ok(state)
    }

    /// Get all open orders from IBKR.
    pub async fn get_open_orders(&self) -> Result<Vec<OpenOrderInfo>> {
        use ibapi::Client;
        use ibapi::orders::Orders;

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 600)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed: {e}"))?;

        let mut subscription = client
            .all_open_orders()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to request open orders: {e}"))?;

        let mut orders = Vec::new();
        let timeout_dur = std::time::Duration::from_secs(10);

        while let Ok(Some(Ok(item))) = tokio::time::timeout(timeout_dur, subscription.next()).await
        {
            match item {
                Orders::OrderData(data) => {
                    orders.push(OpenOrderInfo {
                        order_id: data.order_id,
                        symbol: data.contract.symbol.to_string(),
                        action: format!("{:?}", data.order.action),
                        quantity: data.order.total_quantity,
                        order_type: data.order.order_type.clone(),
                        limit_price: data.order.limit_price,
                        stop_price: data.order.aux_price,
                        status: data.order_state.status.clone(),
                        filled: data.order.filled_quantity,
                        remaining: data.order.total_quantity - data.order.filled_quantity,
                        parent_id: data.order.parent_id,
                    });
                }
                Orders::OrderStatus(_) | Orders::Notice(_) => {}
            }
        }

        Ok(orders)
    }

    /// Cancel a specific order by ID.
    pub async fn cancel_order_by_id(&self, order_id: i32) -> Result<String> {
        use ibapi::Client;
        use ibapi::orders::CancelOrder;

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 700)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed: {e}"))?;

        let mut subscription = client
            .cancel_order(order_id, "")
            .await
            .map_err(|e| anyhow::anyhow!("Failed to cancel order {order_id}: {e}"))?;

        let timeout_dur = std::time::Duration::from_secs(10);
        if let Ok(Some(Ok(item))) = tokio::time::timeout(timeout_dur, subscription.next()).await {
            match item {
                CancelOrder::OrderStatus(status) => return Ok(status.status),
                CancelOrder::Notice(notice) => return Ok(format!("{notice:?}")),
            }
        }

        Ok("cancel_sent".to_string())
    }

    /// Cancel all open orders globally.
    pub async fn cancel_all_orders(&self) -> Result<()> {
        use ibapi::Client;

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 800)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed: {e}"))?;

        client
            .global_cancel()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to cancel all orders: {e}"))?;

        Ok(())
    }
}

#[async_trait]
impl Broker for IbkrBroker {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn check_connection(&self) -> bool {
        use tokio::net::TcpStream;

        matches!(
            tokio::time::timeout(
                std::time::Duration::from_secs(3),
                TcpStream::connect(self.config.connection_url()),
            )
            .await,
            Ok(Ok(_))
        )
    }

    async fn get_price(&self, symbol: &str, asset_class: AssetClass) -> Result<f64> {
        use ibapi::Client;
        use ibapi::contracts::tick_types::TickType;
        use ibapi::market_data::realtime::TickTypes;

        let contract = Self::build_contract(symbol, asset_class)?;

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 200)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed for price check: {e}"))?;

        let mut subscription = client
            .market_data(&contract)
            .snapshot()
            .subscribe()
            .await
            .map_err(|e| anyhow::anyhow!("IBKR snapshot request failed: {e}"))?;

        let timeout_dur = std::time::Duration::from_secs(10);
        let mut best_price: Option<f64> = None;
        let mut bid: Option<f64> = None;
        let mut ask: Option<f64> = None;

        while let Ok(Some(Ok(tick))) = tokio::time::timeout(timeout_dur, subscription.next()).await
        {
            match tick {
                TickTypes::Price(p) if p.price > 0.0 => match p.tick_type {
                    TickType::Last
                    | TickType::Close
                    | TickType::DelayedLast
                    | TickType::DelayedClose => {
                        best_price = Some(p.price);
                    }
                    TickType::Bid | TickType::DelayedBid => bid = Some(p.price),
                    TickType::Ask | TickType::DelayedAsk => ask = Some(p.price),
                    _ => {}
                },
                TickTypes::PriceSize(ps) if ps.price > 0.0 => match ps.price_tick_type {
                    TickType::Last
                    | TickType::Close
                    | TickType::DelayedLast
                    | TickType::DelayedClose => {
                        best_price = Some(ps.price);
                    }
                    TickType::Bid | TickType::DelayedBid => bid = Some(ps.price),
                    TickType::Ask | TickType::DelayedAsk => ask = Some(ps.price),
                    _ => {}
                },
                TickTypes::SnapshotEnd => break,
                _ => {}
            }
        }

        if let Some(price) = best_price {
            return Ok(price);
        }
        if let (Some(b), Some(a)) = (bid, ask) {
            return Ok((b + a) / 2.0);
        }

        anyhow::bail!("No price data received from IBKR snapshot")
    }

    async fn place_order(
        &self,
        symbol: &str,
        asset_class: AssetClass,
        side: Side,
        qty: f64,
    ) -> Result<OrderFill> {
        use ibapi::Client;
        use ibapi::orders::{Action, PlaceOrder, order_builder};

        let contract = Self::build_contract(symbol, asset_class)?;

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 100)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed: {e}"))?;

        let action = match side {
            Side::Buy => Action::Buy,
            Side::Sell => Action::Sell,
        };
        let order = order_builder::market_order(action, qty);

        let order_id = client
            .next_valid_order_id()
            .await
            .map_err(|e| anyhow::anyhow!("failed to get order ID: {e}"))?;

        let mut notifications = client
            .place_order(order_id, &contract, &order)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR order placement failed: {e}"))?;

        let mut filled_qty = 0.0;
        let mut avg_price = 0.0;

        while let Some(result) = notifications.next().await {
            match result {
                Ok(PlaceOrder::ExecutionData(exec)) => {
                    filled_qty = exec.execution.cumulative_quantity;
                    avg_price = exec.execution.average_price;
                }
                Ok(PlaceOrder::CommissionReport(_)) => break,
                Ok(_) => {}
                Err(e) => {
                    warn!("quant: order notification error: {e}");
                    break;
                }
            }
        }

        let filled_usd = filled_qty * avg_price;

        info!(
            "quant: IBKR order filled: {side:?} {filled_qty:.6} {symbol} @ ${avg_price:.2} = ${filled_usd:.2}"
        );

        Ok(OrderFill {
            order_id,
            filled_qty,
            filled_usd,
        })
    }

    async fn get_positions(&self) -> Result<Vec<PositionInfo>> {
        use ibapi::Client;
        use ibapi::accounts::PositionUpdate;

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 400)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed: {e}"))?;

        let mut subscription = client
            .positions()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to request positions: {e}"))?;

        let mut positions = Vec::new();

        while let Some(result) = subscription.next().await {
            match result {
                Ok(PositionUpdate::Position(pos)) => {
                    if pos.position.abs() > f64::EPSILON {
                        positions.push(PositionInfo {
                            account: pos.account.clone(),
                            symbol: pos.contract.symbol.to_string(),
                            security_type: pos.contract.security_type.to_string(),
                            quantity: pos.position,
                            avg_cost: pos.average_cost,
                        });
                    }
                }
                Ok(PositionUpdate::PositionEnd) => break,
                Err(e) => {
                    warn!("quant: position error: {e}");
                    break;
                }
            }
        }

        Ok(positions)
    }

    async fn get_daily_pnl(&self, account: &str) -> Result<DailyPnl> {
        use ibapi::Client;
        use ibapi::accounts::types::AccountId;

        let client = Client::connect(&self.config.connection_url(), self.config.client_id + 500)
            .await
            .map_err(|e| anyhow::anyhow!("IBKR connection failed: {e}"))?;

        let account_id = AccountId(account.to_string());
        let mut subscription = client
            .pnl(&account_id, None)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to request P&L: {e}"))?;

        let timeout_dur = std::time::Duration::from_secs(10);
        match tokio::time::timeout(timeout_dur, subscription.next()).await {
            Ok(Some(Ok(pnl))) => Ok(DailyPnl {
                daily_pnl: pnl.daily_pnl,
                unrealized_pnl: pnl.unrealized_pnl,
                realized_pnl: pnl.realized_pnl,
            }),
            Ok(Some(Err(e))) => anyhow::bail!("P&L error: {e}"),
            Ok(None) => anyhow::bail!("No P&L data received"),
            Err(_) => anyhow::bail!("P&L request timed out"),
        }
    }
}

/// A single price tick from IBKR.
#[derive(Debug, Clone)]
pub struct PriceTick {
    /// Symbol (e.g. "AAPL").
    pub symbol: String,
    /// Last/close price.
    pub price: f64,
    /// Bar volume.
    pub volume: f64,
    /// Tick timestamp (epoch millis).
    pub timestamp: i64,
}

/// Scanner result from IBKR market scanner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    /// Rank in scanner results (0-based).
    pub rank: i32,
    /// Instrument symbol.
    pub symbol: String,
    /// Security type (e.g. "STK", "CRYPTO", "CASH").
    pub security_type: String,
    /// Exchange.
    pub exchange: String,
    /// Currency.
    pub currency: String,
}

/// Info about an open order from IBKR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenOrderInfo {
    /// IBKR order ID.
    pub order_id: i32,
    /// Instrument symbol.
    pub symbol: String,
    /// Side: "BUY" or "SELL".
    pub action: String,
    /// Total order quantity.
    pub quantity: f64,
    /// Order type (e.g. "MKT", "LMT", "STP").
    pub order_type: String,
    /// Limit price (if applicable).
    pub limit_price: Option<f64>,
    /// Stop price (if applicable).
    pub stop_price: Option<f64>,
    /// Order status (e.g. "Submitted", "PreSubmitted", "Filled").
    pub status: String,
    /// Quantity already filled.
    pub filled: f64,
    /// Quantity remaining.
    pub remaining: f64,
    /// Parent order ID (0 if no parent).
    pub parent_id: i32,
}

/// Connect to IBKR and stream price data, sending ticks on the broadcast channel.
///
/// For stocks/crypto: uses `realtime_bars` (5-second OHLCV bars).
/// For forex: uses `tick_by_tick_midpoint` (realtime_bars not supported for CASH contracts).
async fn connect_and_stream(
    symbol: &str,
    config: &IbkrConfig,
    asset_class: AssetClass,
    tx: &broadcast::Sender<PriceTick>,
) -> Result<()> {
    use ibapi::Client;

    let client = Client::connect(&config.connection_url(), config.client_id)
        .await
        .context("failed to connect to IB Gateway")?;

    info!("quant: connected to IB Gateway, subscribing to {symbol}");

    let contract = IbkrBroker::build_contract(symbol, asset_class)?;

    match asset_class {
        AssetClass::Forex => stream_tick_by_tick(symbol, &client, &contract, tx).await,
        _ => stream_realtime_bars(symbol, &client, &contract, tx).await,
    }
}

/// Stream using `realtime_bars` (5-second bars) — works for stocks and crypto.
async fn stream_realtime_bars(
    symbol: &str,
    client: &ibapi::Client,
    contract: &ibapi::contracts::Contract,
    tx: &broadcast::Sender<PriceTick>,
) -> Result<()> {
    use ibapi::market_data::TradingHours;
    use ibapi::market_data::realtime::{BarSize, WhatToShow};

    let mut subscription = client
        .realtime_bars(
            contract,
            BarSize::Sec5,
            WhatToShow::Trades,
            TradingHours::Extended,
        )
        .await
        .context("failed to subscribe to realtime bars")?;

    while let Some(result) = subscription.next().await {
        match result {
            Ok(bar) => {
                let tick = PriceTick {
                    symbol: symbol.to_string(),
                    price: bar.close,
                    volume: bar.volume,
                    timestamp: chrono::Utc::now().timestamp_millis(),
                };

                if tx.send(tick).is_err() {
                    info!("quant: all receivers dropped, stopping feed");
                    return Ok(());
                }
            }
            Err(e) => {
                warn!("quant: error receiving bar: {e}");
                break;
            }
        }
    }

    Ok(())
}

/// Stream using `tick_by_tick_midpoint` — works for forex (CASH contracts on IDEALPRO).
async fn stream_tick_by_tick(
    symbol: &str,
    client: &ibapi::Client,
    contract: &ibapi::contracts::Contract,
    tx: &broadcast::Sender<PriceTick>,
) -> Result<()> {
    let mut subscription = client
        .tick_by_tick_midpoint(contract, 0, false)
        .await
        .context("failed to subscribe to tick-by-tick midpoint")?;

    while let Some(result) = subscription.next().await {
        match result {
            Ok(midpoint) => {
                let tick = PriceTick {
                    symbol: symbol.to_string(),
                    price: midpoint.mid_point,
                    volume: 0.0, // tick-by-tick doesn't include volume
                    timestamp: chrono::Utc::now().timestamp_millis(),
                };

                if tx.send(tick).is_err() {
                    info!("quant: all receivers dropped, stopping feed");
                    return Ok(());
                }
            }
            Err(e) => {
                warn!("quant: error receiving midpoint tick: {e}");
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paper_config() {
        let cfg = IbkrConfig::paper();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 7497);
        assert_eq!(cfg.client_id, 1);
        assert_eq!(cfg.connection_url(), "127.0.0.1:7497");
    }

    #[test]
    fn test_live_config() {
        let cfg = IbkrConfig::live();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 4001);
        assert_eq!(cfg.client_id, 1);
        assert_eq!(cfg.connection_url(), "127.0.0.1:4001");
    }

    #[test]
    fn test_configs_differ() {
        assert_ne!(IbkrConfig::paper().port, IbkrConfig::live().port);
    }

    #[test]
    fn test_price_tick_clone() {
        let tick = PriceTick {
            symbol: "AAPL".into(),
            price: 150.25,
            volume: 1000.0,
            timestamp: 1234567890,
        };
        let cloned = tick.clone();
        assert_eq!(cloned.symbol, "AAPL");
        assert!((cloned.price - 150.25).abs() < f64::EPSILON);
        assert!((cloned.volume - 1000.0).abs() < f64::EPSILON);
        assert_eq!(cloned.timestamp, 1234567890);
    }

    #[tokio::test]
    async fn test_check_connection_unreachable() {
        let broker = IbkrBroker::new("127.0.0.1", 19999, 99);
        assert!(!broker.check_connection().await);
    }

    #[test]
    fn test_build_contract_stock() {
        let contract = IbkrBroker::build_contract("AAPL", AssetClass::Stock).unwrap();
        assert_eq!(contract.symbol.to_string(), "AAPL");
    }

    #[test]
    fn test_build_contract_forex() {
        let contract = IbkrBroker::build_contract("EUR/USD", AssetClass::Forex).unwrap();
        assert_eq!(contract.symbol.to_string(), "EUR");
        assert_eq!(contract.currency.to_string(), "USD");
    }

    #[test]
    fn test_build_contract_forex_invalid() {
        assert!(IbkrBroker::build_contract("EURUSD", AssetClass::Forex).is_err());
    }

    #[test]
    fn test_build_contract_crypto() {
        let contract = IbkrBroker::build_contract("BTC", AssetClass::Crypto).unwrap();
        assert_eq!(contract.symbol.to_string(), "BTC");
    }

    #[test]
    fn test_scan_result_serde() {
        let result = ScanResult {
            rank: 1,
            symbol: "AAPL".into(),
            security_type: "STK".into(),
            exchange: "NASDAQ".into(),
            currency: "USD".into(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let recovered: ScanResult = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.rank, 1);
        assert_eq!(recovered.symbol, "AAPL");
        assert_eq!(recovered.security_type, "STK");
    }

    #[test]
    fn test_open_order_info_serde() {
        let order = OpenOrderInfo {
            order_id: 42,
            symbol: "AAPL".into(),
            action: "Buy".into(),
            quantity: 100.0,
            order_type: "LMT".into(),
            limit_price: Some(185.50),
            stop_price: None,
            status: "Submitted".into(),
            filled: 0.0,
            remaining: 100.0,
            parent_id: 0,
        };
        let json = serde_json::to_string(&order).unwrap();
        let recovered: OpenOrderInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.order_id, 42);
        assert_eq!(recovered.symbol, "AAPL");
        assert_eq!(recovered.status, "Submitted");
        assert!((recovered.remaining - 100.0).abs() < f64::EPSILON);
    }
}
