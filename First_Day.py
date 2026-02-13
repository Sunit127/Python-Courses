# multiple value
# name="Sunit"
# age=21
# print("Name:", name, "Age: ", age)
# print(f"My Name {name} \nMy age is {age}")
"""Data Type"""
# a=1
# b=5.6
# c="Sunit"
# d=True
# print(type(a))
# print(type(b))
# print(type(c))
# print(type(d))
"""Input Ways"""
# name=input("Enter Your Name:")
# print(f"My name is {name}")
"""Operations Core MAth"""
# a=3
# b=7
# print(a+b)
# print(b-a)
# print(b/a)
# print(a*b)
# print(a%b)
# print(b//a)
# print(a**b)
"""If Else -Elif (Decision making)"""

# if age>=18:
#     print("Adult")
# else:
#     print("child")
"""Advance Logic"""
# mark=int(input("Enter Your Mark:"))
# if mark>=90:
#     print("A+")
# elif mark>=80:
#     print("A")
# elif mark>=70:
#     print("B+")
# else:
#     print("Average")
"""Dependent or working"""
# age=int(input("Enter Your Age:"))
# if age >=18 and age<=60:
#     print("Working ")
# if age <13 and age>60:
#     print("Dependent")
"""Mini Project"""
# Name=input("Your Name:")
# Age=int(input("Enter Your Age:"))
# Salary=float(input("Enter Yor Salary:"))
# print(f"Name: {Name}\nAge{Age}\nSalary:{Salary}")
# if Age>=18 and Salary>=50000:
#     print(f"{Name} You are Eligibe For Loan")
# else:
#     print(f"{Name}You are not Eligible for Vote")
"""Day 1 Expert Challenges"""
# Take 3 number and print the largest number
# num1=int(input("Enter a number:"))
# num2=int(input("Enter a number:"))
# num3=int(input("Enter a number:"))
# if num1>num2 and num1>num3:
#     print(f"{num1} is largest number")
# elif num2>num1 and num2>num3:
#     print(f"{num2} is largest number")
# else:
#     print(f"{num3} is largest number")
"""Mini Claculator Beginner with clean version"""
# num1=float(input("Enter a number: "))
# num2=float(input("Enter a number: "))
# print("\nChoose Operations: ")
# print("+ for Addition")
# print("- For Subtraction")
# print("*For multipliucation")
# print("/ for division")
# operation = input("Enter Operation: ")
# # perfome a calculator 
# if operation=="+":
#     print("Result:",num1+num2)
# elif operation=="-":
#     print("Result",num1-num2)
# elif operation=="*":
#     print("Result:,",num1*num2)
# elif operation=="/":
#     if num2 !=0:
#         print("Result: ",num1/num2)
#     else:
#         print(f"Error: Cannot Diviide by {num2}")
# else:
#     print("Invalide Operation selected")
"""Take a number and check positive, negative or zero"""
num=float(input("Enter a number: "))
if num<0:
    print(f"{num} is negative")
elif num>0:
    print(f"{num} is postive")
else:
    print(f"number is {num}")