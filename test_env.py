from env import StartupEnv
import gymnasium as gym

def test_startup_env():
    env = StartupEnv()
    obs, info = env.reset()
    print("Initial State:", obs)

    # Manual sequence of actions: hire, build, market, raise, do_nothing
    actions = [0, 2, 3, 4, 5, 1]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action} | Month: {obs['month'][0]} | Cash: {obs['cash'][0]:.0f} | Revenue: {obs['revenue'][0]:.0f} | Quality: {obs['product_quality'][0]:.2f} | Growth: {reward:.2f}")
        print(f"Events: {info['events']}")
        
        if terminated or truncated:
            break

    print("Test passed successfully.")

if __name__ == "__main__":
    test_startup_env()
