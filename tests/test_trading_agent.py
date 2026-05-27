import unittest

from rl_trading_agent import Action, TradingEnvironment, train_trading_agent
from rl_trading_agent.environment import State

STARTING_CASH = 1_000.0


class TradingEnvironmentTest(unittest.TestCase):
    def test_buy_then_hold_tracks_price_increase(self) -> None:
        environment = TradingEnvironment([10, 12, 13], starting_cash=100)

        state = environment.reset()
        self.assertEqual(state, State(trend=0, holding=0))

        next_state, reward, done, info = environment.step(Action.BUY)
        self.assertEqual(next_state, State(trend=1, holding=1))
        self.assertEqual(reward, 2.0)
        self.assertFalse(done)
        self.assertEqual(info["portfolio_value"], 102.0)

        next_state, reward, done, info = environment.step(Action.HOLD)
        self.assertEqual(next_state, State(trend=1, holding=1))
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        self.assertEqual(info["portfolio_value"], 103.0)

    def test_invalid_price_series_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TradingEnvironment([100])

        with self.assertRaises(ValueError):
            TradingEnvironment([100, 0])


class QLearningTraderTest(unittest.TestCase):
    def test_training_populates_q_table_and_learns_profitable_path(self) -> None:
        trader, summary = train_trading_agent([10, 11, 12, 13, 14], episodes=25, seed=7)

        self.assertTrue(trader.q_table)
        self.assertIn("portfolio_value", summary)
        self.assertGreater(summary["portfolio_value"], STARTING_CASH)
        upward_flat_state = State(trend=1, holding=0)
        self.assertGreater(
            trader.q_table[upward_flat_state][Action.BUY],
            trader.q_table[upward_flat_state][Action.HOLD],
        )


if __name__ == "__main__":
    unittest.main()
