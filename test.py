def day_profit():
    capital = 50
    profit = 0.020   
    days = 365

    print(f"{'Day':>5} | {'Capital ($)':>15}")
    print("-" * 25)

    for i in range(1, days + 1):
        if i % 30 == 0:
            capital += 100
        capital *= (1 + profit)
        print(f"{i:>5} | {capital:>15.2f}$")
    print("-" * 34)
    print(f"FINAL: {capital:.2f}$")
    print()




def trading_no_percent():
    capital = 100.0
    lot = 0.01
    days = 365
    orders_per_day = 10

    print(f"{'Day':>4} | {'Capital($)':>12} | {'Lot':>5} | {'Daily Profit':>12}")
    print("-" * 42)

    for day in range(1, days + 1):
        profit_per_order = lot * 100    
        daily_profit = orders_per_day * profit_per_order
        capital += daily_profit

        lot = 0.01 + int(capital // 100) * 0.01

        print(f"{day:>4} | {capital:>12.2f} | {lot:>5.2f} | {daily_profit:>12.2f}")

    print("\nRESULT:")
    print(f"Final capital: {capital:.2f}$")
    print(f"Final lot: {lot:.2f}")


trading_no_percent()






