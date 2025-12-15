from signals import get_long_short_ids
from trades import execute_trades, shutdown

def main():
    long_ids, short_ids = get_long_short_ids(n=3)
    execute_trades(long_ids, short_ids)
    shutdown()

if __name__ == "__main__":
    main()
