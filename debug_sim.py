from app import SimulatorUI

def test_sim():
    print("Initializing Simulator...")
    sim = SimulatorUI()
    print("Resetting Simulator...")
    try:
        state, plot_data, logs = sim.reset_sim()
        print("Initial State Info:", state)
    except Exception as e:
        print(f"FAILED on reset_sim: {e}")
        import traceback
        traceback.print_exc()
        return

    actions = ["Hire Engineer", "Build Feature", "Run Marketing", "Raise Funding", "Do Nothing", "Pivot", "Train Team"]
    for action in actions:
        print(f"Testing manual action: {action}")
        try:
            result = sim.step_manual(action)
            print(f"Action {action} result: {result[0]}")
        except Exception as e:
            print(f"FAILED on step_manual ({action}): {e}")
            import traceback
            traceback.print_exc()
            break
            
    print("Testing Automated Agent...")
    try:
        sim.run_agent("Rule-based")
        print("Rule-based simulation complete.")
    except Exception as e:
        print(f"FAILED on run_agent (Rule-based): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sim()
