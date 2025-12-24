# Simple program to take two numbers as input and return their sum

def main():
    try:
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
        total = num1 + num2
        print(f"The sum of {num1} and {num2} is {total}.")
    except ValueError:
        print("Please enter valid numbers.")

if __name__ == "__main__":
    main()