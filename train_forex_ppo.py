from __future__ import annotations

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from forex_rl_trading.envs.forex import ForexTradingEnv
from forex_rl_trading.features import prepare_features

DATA_FILE = "data/EURUSD_2024-01-01_to_2024-12-31_after_dt_transfo.csv"

TRAIN_TICKS = 500_000
EVAL_TICKS = 100_000
TOTAL_TIMESTEPS = 500_000

INITIAL_BALANCE = 10_000.0
TP = 0.0002
SL = 0.0002
COMMISSION_PER_STD_LOT_PER_SIDE = 0.0
SPREAD_COST_PER_STD_LOT_PER_SIDE = 0.0
POSITION_SIZE = 100_000
LOOKBACK_WINDOW = 1_000
NEG_REWARD_FOR_WAITING = -0.0001

DEVICE = "cpu"
SEED = 42


def make_env(df_slice: pd.DataFrame):
    def _make():
        env = ForexTradingEnv(
            df=df_slice.copy(),
            initial_balance=INITIAL_BALANCE,
            tp=TP,
            sl=SL,
            commission_per_std_lot_per_side=COMMISSION_PER_STD_LOT_PER_SIDE,
            spread_cost_per_std_lot_per_side=SPREAD_COST_PER_STD_LOT_PER_SIDE,
            position_size=POSITION_SIZE,
            lookback_window=LOOKBACK_WINDOW,
            neg_reward_for_waiting=NEG_REWARD_FOR_WAITING,
        )
        return Monitor(env)

    return _make


print("Read data...")
df = pd.read_csv(DATA_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

print("Prepare features...")
df = prepare_features(df)
print(f"df.shape = {df.shape}")

df_train = df.iloc[:TRAIN_TICKS].copy()
df_eval = df.iloc[TRAIN_TICKS : TRAIN_TICKS + EVAL_TICKS].copy()
print(f"Train ticks: {len(df_train)} | Eval ticks: {len(df_eval)}")

if len(df_train) < 10_000:
    raise ValueError("Not enough training ticks to run PPO sanely.")
if len(df_eval) < 1_000:
    raise ValueError("Not enough eval ticks to evaluate.")

train_env = DummyVecEnv([make_env(df_train)])
eval_env = DummyVecEnv([make_env(df_eval)])
underlying_env = eval_env.envs[0].env  # Monitor -> ForexTradingEnv

model = PPO("MlpPolicy", train_env, verbose=1, device=DEVICE, seed=SEED)

print("\nTraining...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
print("Training completed.")

print("\n" + "=" * 60)
print("EVALUATION (FROZEN POLICY)")
print("=" * 60)

obs = eval_env.reset()
action_counts = {0: 0, 1: 0, 2: 0}

for i in range(len(df_eval)):
    action, _ = model.predict(obs, deterministic=True)

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs).to(model.device)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]

    a = int(action[0])
    action_counts[a] = action_counts.get(a, 0) + 1

    if i < 10:
        print(f"[First actions] step={i:02d} action={a} probs={probs}")

    obs, reward, done, info = eval_env.step(action)

    if i % 5000 == 0:
        s = underlying_env.get_stats()
        print(
            f"[Eval step {i}] balance={s['balance']:.2f} profit={s['profit']:.2f} "
            f"trades={s['n_trades']} closed={s['n_closed']}"
        )

    if done[0]:
        print(f"Episode ended at step {i}")
        break

final_stats = underlying_env.get_stats()
total_actions = sum(action_counts.values())

print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)
print(f"Balance: {final_stats['balance']:.2f}")
print(f"Profit:  {final_stats['profit']:.2f}")
print(f"Trades:  {final_stats['n_trades']} (closed: {final_stats['n_closed']})")

print("\nAction distribution:")
for k, name in [(0, "HOLD"), (1, "BUY "), (2, "SELL")]:
    c = action_counts.get(k, 0)
    pct = (c / total_actions * 100.0) if total_actions > 0 else 0.0
    print(f"{name}: {c} ({pct:.1f}%)")
