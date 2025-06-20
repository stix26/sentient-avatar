import time

import psycopg2
from psycopg2 import OperationalError

MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds


def connect_with_retry() -> None:
    """Attempt to connect to PostgreSQL, creating the role if necessary."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = psycopg2.connect(
                dbname="postgres",
                user="postgres",
                password="postgres",
                host="postgres",
                port=5432,
            )
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", ("postgres",))
                if cur.fetchone() is None:
                    cur.execute("CREATE ROLE postgres LOGIN SUPERUSER PASSWORD %s", ("postgres",))
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", ("test_db",))
                if cur.fetchone() is None:
                    cur.execute("CREATE DATABASE test_db OWNER postgres")
            conn.close()

            conn = psycopg2.connect(
                dbname="test_db",
                user="postgres",
                password="postgres",
                host="postgres",
                port=5432,
            )
            print("Database connection established.")
            conn.close()
            return
        except OperationalError as exc:
            print(f"Attempt {attempt} failed: {exc}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY)


if __name__ == "__main__":
    connect_with_retry()
