capital = 100
days = 365

print(f"{'Day':>5} | {'Capital ($)':>15}")
print("-" * 25)


for day in range(1, days + 1):

    # Ҳар 30 рӯз 30$ илова
    if day % 30 == 0:
        capital += 30

    # Фоизи афзоиш
    if day <= 30:
        rate = 0.05
    elif day <= 50:
        rate = 0.03
    elif day <= 250:
        rate = 0.02
    else:
        rate = 0.01

    capital *= (1 + rate)

    print(f"{day:>5} | {capital:>15.2f}")

print("-" * 25)
print(f"FINAL CAPITAL: {capital:.2f}$")
# FINAL CAPITAL: 147331.06$