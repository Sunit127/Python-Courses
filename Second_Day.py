"""For Loop"""
# for i in range(0,5):
#     print(i)
"""Using loop to print 1 to 10"""
# for i in range(1,11):
#     print(i)
"""Using For loop to print Reverse to print 10 to 1"""
# for i in range(10,1,-1):
#     print(i)
"""While Loop Conditiom Base Loop"""
# i=1
# while i<=5:
#     print(i)
#     i+=1
"""Using Break"""
# for i in range(1,10):
#     if i==6:
#         # it will stop on 5
#         break 
#     print(i)
# for i in range(1,10):
#     if i==7:
#         # skip 7 only
#         continue
#     print(i)
"""Sum using For Loop"""
# total=0
# for i in range(1,10):
#     total +=i
# print("Sum",total)
"""Sum using while Loop"""
# total = 0
# i = 1

# while i <= 10:
#     total += i
#     i += 1

# print("Sum =", total)
"""Even Num from 1 to 20"""
# for i in range(1,21):
#     # for odd just change the 0 into 1 
#     if i%2==0:
#         print(i)
"""Nested Loop"""
# for i in range(3):
#     for j in range(3):
#         print(i,j)
"""Now Star Pattern Program"""
# for i in range(1,11):
#     print("A"*i)
"""Number Triangle"""
# for i in range(1,9):
#     for j in range(1,i+1):
#         print(j,end="")
#     print()
"""DAY 2 Practice Question"""
# Print odd number from 1-50
# for i in range(1,52):
#     if i%2==1:
# #         print(i) 
"""Using While Loop"""
# i=1
# while i<=50:
#     if i%2==1:
#         print(i)
#     i+=1
"""Find Factorial Number of a number"""
# num=5
# i=1
# factorial=1
# while i<=num:
#     factorial*=i
#     i+=1
# print(factorial)
"""Reverse a number"""
# for i in range(10,1,-1):
#     print(i)
# i=8
# while i>=1:
#     print(i)
#     i-=1
"""Count How Many digits in a number"""
# num = int(input("Enter number: "))

# count = 0

# while num > 0:
#     num = num // 10
#     count += 1

# print("Total digits =", count)
num=2
for i in range(1,11):
    print(f"{num}x{i}={num*i}")

    