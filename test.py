day_30 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

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

        if i in day_30:
            capital *= (1 + profit30) 

        else:
            capital *= (1 + profit)
        print(f"{i:>5} | {capital:>15.2f}$")
    print("-" * 34)
    print(f"FINAL: {capital:.2f}$")
    print()


main()


