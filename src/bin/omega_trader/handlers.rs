//! CLI command handlers for omega-trader subcommands.

use omega_trader::broker::{AssetClass, build_broker};
use omega_trader::execution::{ImmediatePlan, Side};
use omega_trader::executor::{
    CircuitBreaker, DailyLimits, Executor, check_daily_pnl_cutoff, check_max_positions,
};

/// Print a JSON connectivity error and exit.
pub fn connectivity_error(host: &str, port: u16) -> ! {
    let err = serde_json::json!({
        "error": format!("Broker not reachable at {host}:{port}"),
    });
    println!("{}", serde_json::to_string(&err).unwrap());
    std::process::exit(1);
}

/// Parse a side string into a `Side` enum.
pub fn parse_side(s: &str) -> anyhow::Result<Side> {
    match s.to_lowercase().as_str() {
        "buy" => Ok(Side::Buy),
        "sell" => Ok(Side::Sell),
        _ => anyhow::bail!("Invalid side '{s}'. Use 'buy' or 'sell'."),
    }
}

/// Handle the `check` subcommand.
pub async fn handle_check(broker_name: &str, host: String, port: u16) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    let connected = broker.check_connection().await;
    let result = serde_json::json!({
        "connected": connected,
        "host": host,
        "port": port,
    });
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

/// Handle the `scan` subcommand.
#[allow(clippy::too_many_arguments)]
pub async fn handle_scan(
    broker_name: &str,
    scan_code: &str,
    instrument: &str,
    location: &str,
    count: i32,
    min_price: Option<f64>,
    min_volume: Option<i32>,
    host: String,
    port: u16,
) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    if !broker.check_connection().await {
        connectivity_error(&host, port);
    }

    // Downcast to IbkrBroker for scanner (IBKR-specific feature).
    #[cfg(feature = "ibkr")]
    {
        if let Some(ibkr) = broker
            .as_any()
            .downcast_ref::<omega_trader::broker::ibkr::IbkrBroker>()
        {
            let results = ibkr
                .run_scanner(
                    scan_code, instrument, location, count, min_price, min_volume,
                )
                .await?;
            println!("{}", serde_json::to_string(&results)?);
            return Ok(());
        }
    }

    anyhow::bail!("Scanner is only supported for the IBKR broker");
}

/// Handle the `analyze` subcommand.
#[allow(clippy::too_many_arguments)]
pub async fn handle_analyze(
    broker_name: &str,
    symbol: &str,
    asset_class: &str,
    portfolio: f64,
    host: String,
    port: u16,
    bars: u32,
) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    if !broker.check_connection().await {
        connectivity_error(&host, port);
    }

    let parsed_class: AssetClass = asset_class.parse()?;

    // Downcast to IbkrBroker for price feed (IBKR-specific feature).
    #[cfg(feature = "ibkr")]
    {
        if let Some(ibkr) = broker
            .as_any()
            .downcast_ref::<omega_trader::broker::ibkr::IbkrBroker>()
        {
            let mut engine = omega_trader::QuantEngine::new(symbol, portfolio);
            let mut rx = ibkr.start_price_feed(symbol, parsed_class);
            let mut count: u32 = 0;

            // Timeout for first bar — if no data within 15s, the market is likely closed.
            let first = tokio::time::timeout(std::time::Duration::from_secs(15), rx.recv()).await;
            match first {
                Ok(Ok(tick)) => {
                    let signal = engine.process_price(tick.price);
                    println!("{}", serde_json::to_string(&signal)?);
                    count += 1;
                }
                _ => {
                    let err = serde_json::json!({
                        "error": format!("No data received for {symbol} ({parsed_class}) within 15s — market may be closed or data subscription missing"),
                    });
                    println!("{}", serde_json::to_string(&err)?);
                    std::process::exit(1);
                }
            }

            while count < bars {
                match tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv()).await {
                    Ok(Ok(tick)) => {
                        let signal = engine.process_price(tick.price);
                        println!("{}", serde_json::to_string(&signal)?);
                        count += 1;
                    }
                    _ => break,
                }
            }
            return Ok(());
        }
    }

    anyhow::bail!("Analyze (price feed) is only supported for the IBKR broker");
}

/// Handle the `order` subcommand.
#[allow(clippy::too_many_arguments)]
pub async fn handle_order(
    broker_name: &str,
    symbol: &str,
    side: &str,
    quantity: f64,
    asset_class: &str,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    account: Option<&str>,
    portfolio: Option<f64>,
    max_positions: usize,
    host: String,
    port: u16,
) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    if !broker.check_connection().await {
        connectivity_error(&host, port);
    }

    let order_side = parse_side(side)?;
    let parsed_class: AssetClass = asset_class.parse()?;

    // Safety: check position count.
    if let Ok(positions) = broker.get_positions().await {
        check_max_positions(positions.len(), max_positions)?;
    }

    // Safety: check daily P&L cutoff.
    if let (Some(acct), Some(port_val)) = (account, portfolio)
        && let Ok(pnl) = broker.get_daily_pnl(acct).await
    {
        check_daily_pnl_cutoff(pnl.daily_pnl, port_val, 5.0)?;
    }

    if let (Some(sl_pct), Some(tp_pct)) = (stop_loss, take_profit) {
        // Bracket order: fetch entry price, calculate SL/TP levels.
        let entry_price = broker.get_price(symbol, parsed_class).await?;
        let (sl_price, tp_price) = match order_side {
            Side::Buy => (
                entry_price * (1.0 - sl_pct / 100.0),
                entry_price * (1.0 + tp_pct / 100.0),
            ),
            Side::Sell => (
                entry_price * (1.0 + sl_pct / 100.0),
                entry_price * (1.0 - tp_pct / 100.0),
            ),
        };

        // Downcast to IbkrBroker for bracket orders (IBKR-specific feature).
        #[cfg(feature = "ibkr")]
        {
            if let Some(ibkr) = broker
                .as_any()
                .downcast_ref::<omega_trader::broker::ibkr::IbkrBroker>()
            {
                let contract =
                    omega_trader::broker::ibkr::IbkrBroker::build_contract(symbol, parsed_class)?;
                let state = ibkr
                    .place_bracket_order(&contract, order_side, quantity, tp_price, sl_price)
                    .await?;

                let result = serde_json::json!({
                    "type": "bracket",
                    "status": format!("{:?}", state.status),
                    "entry_price": entry_price,
                    "stop_loss_price": sl_price,
                    "take_profit_price": tp_price,
                    "filled_qty": state.total_filled_qty,
                    "filled_usd": state.total_filled_usd,
                    "order_ids": state.order_ids,
                    "errors": state.errors,
                });
                println!("{}", serde_json::to_string(&result)?);
                return Ok(());
            }
        }

        anyhow::bail!("Bracket orders are only supported for the IBKR broker");
    } else {
        // Simple market order via Broker trait.
        let plan = omega_trader::execution::ExecutionPlan::Immediate(ImmediatePlan {
            symbol: symbol.to_string(),
            side: order_side,
            quantity,
            estimated_price: 0.0,
            estimated_usd: 0.0,
            asset_class: parsed_class,
        });

        let circuit_breaker = CircuitBreaker::default();
        let daily_limits = DailyLimits::new(10, 50_000.0, 5);
        let mut executor = Executor::new(
            build_broker(broker_name, &host, port, 1)?,
            circuit_breaker,
            daily_limits,
        );
        let state = executor.execute(&plan).await;

        let result = serde_json::json!({
            "type": "market",
            "status": format!("{:?}", state.status),
            "filled_qty": state.total_filled_qty,
            "avg_price": if state.total_filled_qty > 0.0 {
                state.total_filled_usd / state.total_filled_qty
            } else {
                0.0
            },
            "filled_usd": state.total_filled_usd,
            "errors": state.errors,
            "abort_reason": state.abort_reason,
        });
        println!("{}", serde_json::to_string(&result)?);
    }
    Ok(())
}

/// Handle the `positions` subcommand.
pub async fn handle_positions(broker_name: &str, host: String, port: u16) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    if !broker.check_connection().await {
        connectivity_error(&host, port);
    }

    let positions = broker.get_positions().await?;
    println!("{}", serde_json::to_string(&positions)?);
    Ok(())
}

/// Handle the `pnl` subcommand.
pub async fn handle_pnl(
    broker_name: &str,
    account: &str,
    host: String,
    port: u16,
) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    if !broker.check_connection().await {
        connectivity_error(&host, port);
    }

    let pnl = broker.get_daily_pnl(account).await?;
    println!("{}", serde_json::to_string(&pnl)?);
    Ok(())
}

/// Handle the `close` subcommand.
pub async fn handle_close(
    broker_name: &str,
    symbol: &str,
    asset_class: &str,
    quantity: Option<f64>,
    host: String,
    port: u16,
) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    if !broker.check_connection().await {
        connectivity_error(&host, port);
    }

    let parsed_class: AssetClass = asset_class.parse()?;

    // Determine side and quantity from current position.
    let positions = broker.get_positions().await?;
    let match_symbol = match parsed_class {
        AssetClass::Forex => symbol.split('/').next().unwrap_or(symbol).to_string(),
        _ => symbol.to_string(),
    };
    let pos = positions
        .iter()
        .find(|p| p.symbol == match_symbol)
        .ok_or_else(|| anyhow::anyhow!("No open position found for {symbol}"))?;

    let close_qty = quantity.unwrap_or(pos.quantity.abs());
    let close_side = if pos.quantity > 0.0 {
        Side::Sell
    } else {
        Side::Buy
    };

    // Downcast to IbkrBroker for close_position (IBKR-specific).
    #[cfg(feature = "ibkr")]
    {
        if let Some(ibkr) = broker
            .as_any()
            .downcast_ref::<omega_trader::broker::ibkr::IbkrBroker>()
        {
            let contract =
                omega_trader::broker::ibkr::IbkrBroker::build_contract(symbol, parsed_class)?;
            let state = ibkr
                .close_position(&contract, close_qty, close_side)
                .await?;

            let result = serde_json::json!({
                "status": format!("{:?}", state.status),
                "side": format!("{close_side}"),
                "closed_qty": state.total_filled_qty,
                "filled_usd": state.total_filled_usd,
                "errors": state.errors,
            });
            println!("{}", serde_json::to_string(&result)?);
            return Ok(());
        }
    }

    anyhow::bail!("Close position is only supported for the IBKR broker");
}

/// Handle the `orders` subcommand.
pub async fn handle_orders(broker_name: &str, host: String, port: u16) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    if !broker.check_connection().await {
        connectivity_error(&host, port);
    }

    // Downcast to IbkrBroker for open orders (IBKR-specific).
    #[cfg(feature = "ibkr")]
    {
        if let Some(ibkr) = broker
            .as_any()
            .downcast_ref::<omega_trader::broker::ibkr::IbkrBroker>()
        {
            let orders = ibkr.get_open_orders().await?;
            println!("{}", serde_json::to_string(&orders)?);
            return Ok(());
        }
    }

    anyhow::bail!("Open orders listing is only supported for the IBKR broker");
}

/// Handle the `cancel` subcommand.
pub async fn handle_cancel(
    broker_name: &str,
    order_id: Option<i32>,
    host: String,
    port: u16,
) -> anyhow::Result<()> {
    let broker = build_broker(broker_name, &host, port, 1)?;
    if !broker.check_connection().await {
        connectivity_error(&host, port);
    }

    // Downcast to IbkrBroker for cancel (IBKR-specific).
    #[cfg(feature = "ibkr")]
    {
        if let Some(ibkr) = broker
            .as_any()
            .downcast_ref::<omega_trader::broker::ibkr::IbkrBroker>()
        {
            if let Some(id) = order_id {
                let status = ibkr.cancel_order_by_id(id).await?;
                let result = serde_json::json!({
                    "cancelled": id,
                    "status": status,
                });
                println!("{}", serde_json::to_string(&result)?);
            } else {
                ibkr.cancel_all_orders().await?;
                let result = serde_json::json!({
                    "cancelled": "all",
                    "status": "global_cancel_sent",
                });
                println!("{}", serde_json::to_string(&result)?);
            }
            return Ok(());
        }
    }

    anyhow::bail!("Order cancellation is only supported for the IBKR broker");
}
