lst = list(map(int, input().split()))
mx = -9999

for i in lst:
    if i > mx:
        mx = i


print(mx)