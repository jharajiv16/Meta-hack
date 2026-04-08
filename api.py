from fastapi import FastAPI
from env import StartupEnv

app = FastAPI()

env = StartupEnv()


@app.post("/reset")
def reset():
    state, info = env.reset()
    return {"state": state}


@app.post("/step")
def step(action: int):
    obs, reward, terminated, truncated, info = env.step(action)
    return {
        "state": obs,
        "reward": reward,
        "done": terminated or truncated
    }