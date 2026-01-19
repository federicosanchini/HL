# main.py
from signals import get_long_short_ids
from trades import execute_trades, shutdown
from close_trades import close_net_in_time_window  # <-- importa la funzione di close
import os
import time


# =====================
# CONFIG CLOSE WINDOW
# =====================
CLOSE_MIN_AGE = 29.9
CLOSE_MAX_AGE = 30.1
CLOSE_UNIT = "days"
CLOSE_CLAMP = True


def main():
    try:
        # 1) Chiudi le size delle coin tradeate nella finestra temporale
        '''
        close_net_in_time_window(
            min_age=CLOSE_MIN_AGE,
            max_age=CLOSE_MAX_AGE,
            unit=CLOSE_UNIT,
            clamp_to_position=CLOSE_CLAMP,
        )
        '''

        # 2) Apri/gestisci trade come fai già
        long_ids, short_ids = get_long_short_ids(n=3)
        execute_trades(long_ids, short_ids)

    finally:
        time.sleep(10)

        # 3) Shutdown sempre (anche se sopra va in errore)
        shutdown()

        # Se vuoi mantenere l’uscita “brutale” come prima:
        os._exit(0)


if __name__ == "__main__":
    main()
