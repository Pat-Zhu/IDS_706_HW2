import argparse, json
from .stock_utils import run_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-01-01")
    args = p.parse_args()
    out = run_pipeline(args.ticker, args.start, args.end)
    print("=== Pipeline Summary ===")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
