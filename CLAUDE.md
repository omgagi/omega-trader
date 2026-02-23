# CLAUDE.md — omega-trader

## Project

Standalone, broker-agnostic quantitative trading engine. Kalman filter, HMM regime detection, fractional Kelly sizing, Merton allocation, IBKR execution. Originally extracted from the `omega` monorepo.

**Repository:** `github.com/omgagi/omega-trader`

## Build & Test

```bash
cargo build                  # Build with default features (ibkr)
cargo clippy -- -D warnings  # Zero warnings required
cargo test                   # All tests must pass
cargo fmt --check            # Format check
cargo build --release        # Optimized binary
```

## Architecture

Single crate with broker abstraction:

| Module | Purpose |
|--------|---------|
| `lib.rs` | `QuantEngine` orchestrator — Kalman + HMM + Merton + Kelly pipeline |
| `broker/mod.rs` | `Broker` trait (5 methods), shared types (`AssetClass`, `OrderFill`, `PositionInfo`, `DailyPnl`) |
| `broker/ibkr.rs` | `IbkrBroker` — IBKR TWS API implementation (gated by `ibkr` feature) |
| `executor/mod.rs` | `Executor` using `Box<dyn Broker>` — circuit breaker, daily limits, TWAP/Immediate |
| `executor/guardrails.rs` | Safety checks, state persistence, execution reporting |
| `signal.rs` | `QuantSignal`, `Regime`, `Direction`, `Action`, `ExecutionStrategy` |
| `kalman.rs` | 2D Kalman filter (price + trend) |
| `hmm.rs` | 3-state HMM with Baum-Welch training |
| `kelly.rs` | Fractional Kelly criterion with safety clamps |
| `execution.rs` | TWAP + Immediate execution plan types |
| `bin/omega_trader/` | CLI binary with 9 subcommands + `--broker` flag |

## Feature Flags

- `ibkr` (default) — enables IBKR TWS API via `ibapi` crate
- Without `ibkr`: only the quant engine library compiles (no broker)

## Key Design Rules

- **No `unwrap()`** — use `?` and proper error types
- **Broker trait for abstraction** — concrete broker-specific methods stay on the impl, not the trait
- **Feature-gated deps** — `ibapi` is optional via `#[cfg(feature = "ibkr")]`
- **Downcast for broker-specific** — handlers use `as_any()` + downcast for scanner, bracket orders, etc.
