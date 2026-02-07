
def main():
    capital = 100
    profit = 0.020
    profit30 = 0.040   
    days = 365


    print(f"{'Day':>5} | {'Capital ($)':>15}")
    print("-" * 25)

    for i in range(1, days + 1):

        if i % 30 == 0:
            capital += 50 

        if i <= 40:
            capital *= (1 + profit30)
        elif i > 40 and i <= 250:
            capital *= (1 + profit)
        else:
            capital *= (1 + 0.01)
        print(f"{i:>5} | {capital:>15.2f}$")

    print("-" * 34)
    print(f"FINAL: {capital:.2f}$")
    print()


main()
