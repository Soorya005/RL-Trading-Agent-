# Trading Environment Design (Phase 2)

## What the environment represents
The environment simulates a single-asset trading loop where the agent decides to buy, sell, or hold each step.

## State (observation)
- Price and indicators: Close, RSI, MACD, MACD_Signal, MA_10, MA_50
- Portfolio context: cash balance and current position

## Action space
- 0: Hold
- 1: Buy 1 unit
- 2: Sell 1 unit

## Reward
- Profit and loss (change in net worth)
- Risk penalty based on drawdown

## Episode termination
- Ends when the data runs out

## Common beginner mistakes
- Ignoring transaction costs
- Using future data by mistake
- Rewarding only raw profit without risk penalty
