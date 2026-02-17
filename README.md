📘 Python Mastery Journey – Day 2
🔁 Loops & Control Flow Practice
🚀 Overview

This repository contains my Day 2 Python practice, focused on mastering:

for loops

while loops

range() function

break & continue

Nested loops

Pattern programs

Real-world logic problems

The goal of Day 2 was to build strong logical thinking and repetition control using loops.

🧠 Topics Covered
🔹 1. For Loop

Used to repeat code a fixed number of times.

for i in range(1, 11):
    print(i)

🔹 2. While Loop

Used when repetition depends on a condition.

i = 1
while i <= 5:
    print(i)
    i += 1

🔹 3. Break & Continue
# Break example
for i in range(10):
    if i == 5:
        break
    print(i)

# Continue example
for i in range(10):
    if i == 5:
        continue
    print(i)

🔹 4. Multiplication Table Program
num = int(input("Enter a number: "))

for i in range(1, 11):
    print(f"{num} x {i} = {num * i}")

🔹 5. Factorial Program
num = int(input("Enter a number: "))
fact = 1

for i in range(1, num + 1):
    fact *= i

print("Factorial:", fact)

🔹 6. Number Triangle Pattern
for i in range(1, 6):
    for j in range(1, i + 1):
        print(j, end=" ")
    print()

🎯 Key Learning Outcomes

✔ Understanding iteration
✔ Writing clean loop logic
✔ Solving mathematical problems
✔ Building repetition-based systems
✔ Improving logical thinking

💡 Why This Matters

Loops are the backbone of:

Automation

Data processing

AI models

Game development

Backend systems

Mastering loops means mastering control over repetition and logic flow.

🛠 Technologies Used

Python 3.x

VS Code

Git & GitHub

📈 Progress

✅ Day 1 – Basics & Conditionals

✅ Day 2 – Loops & Control Flow

🔜 Day 3 – Functions & Lists

👨‍💻 Author

Sunit Sah
BSc (Hons) Computer Systems Engineering
Python Mastery Journey 🚀

🔥 Pro Tip

After uploading:

Commit message: Completed Day 2 – Loop Mastery

Push to GitHub

Keep consistency daily
