def main():
    capital = 100
    profit = 0.020   
    days = 365

    print(f"{'Day':>5} | {'Capital ($)':>15}")
    print("-" * 25)

    for i in range(1, days + 1):
        if i % 30 == 0:
            capital += 50
        capital *= (1 + profit)
        print(f"{i:>5} | {capital:>15.2f}$")
    print("-" * 34)
    print(f"FINAL: {capital:.2f}$")
    print()


main()
