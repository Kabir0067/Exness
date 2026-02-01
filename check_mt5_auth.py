
import os
import sys
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Force load .env
load_dotenv(override=True)

def check():
    login_str = os.getenv("EXNESS_LOGIN", "")
    password = os.getenv("EXNESS_PASSWORD", "")
    server = os.getenv("EXNESS_SERVER", "")

    print(f"--- MT5 Credential Check ---")
    print(f"ENV 'EXNESS_LOGIN':    '{login_str}'")
    print(f"ENV 'EXNESS_SERVER':   '{server}'")
    print(f"ENV 'EXNESS_PASSWORD': {'(SET, len=' + str(len(password)) + ')' if password else '(EMPTY)'}")

    if not login_str or not password or not server:
        print("\nERROR: Missing credentials in .env file.")
        return

    try:
        login = int(login_str)
    except ValueError:
        print(f"\nERROR: Login must be an integer. Got '{login_str}'")
        return

    print(f"\nAttempting MT5 Init...")
    if not mt5.initialize(login=login, password=password, server=server):
        code, msg = mt5.last_error()
        print(f"FAILED: data={code}, description={msg}")
        
        # Extended help
        if code == -6: # Authorization failed
            print("\nSUGGESTION: This is an 'Authorization failed' error.")
            print("1. double check your trading password (not investor password).")
            print("2. Check if Server is exactly correct (e.g. 'Exness-MT5Real' vs 'Exness-Real1').")
            print("3. Check if Account Number is correct.")
    else:
        print("\nSUCCESS: Authorization working!")
        ai = mt5.account_info()
        if ai:
            print(f"Account: {ai.login}, Name: {ai.name}, Balance: {ai.balance}, Server: {ai.server}")
        mt5.shutdown()

if __name__ == "__main__":
    check()
