deposit = 100
days = 250
profit = 0.03 


print('-'*40)
print(f'|{"Day":>14} | {"Capital":>20}|') 
print('-'*40)

for i in range(1, days + 1):
    deposit *= (1 + profit)
    if i % 10 == 0:
        print(f'|Day - {i:>4} | capital {deposit:>15,.2f}|')

print('-'*40)