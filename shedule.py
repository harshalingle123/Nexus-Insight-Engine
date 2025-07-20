import time
import threading
import schedule
import subprocess

# ──────────────────────────────────────────────────────────────
MARKET_RUN_TIME = "09:00"  # Adjust to before market opens (24h format)
# ──────────────────────────────────────────────────────────────

def ingest_job():
    try:
        print("Starting: market_clening.py")
        subprocess.run(["python", "market_clening.py"], check=True)
        print("Completed: market_clening.py")

        print("Starting: feature_engineering.py")
        subprocess.run(["python", "feature_engineering.py"], check=True)
        print("Completed: feature_engineering.py")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def start_scheduler():
    schedule.every().day.at(MARKET_RUN_TIME).do(ingest_job)
    print(f"Scheduler set to run at {MARKET_RUN_TIME} daily.")

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=start_scheduler, daemon=True).start()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Scheduler stopped.")
