from sympy import symbols, diff, sympify, exp

# Quiz1. 다항식을 위한 미분 계산기
def differentiate_polynomial(expression):
    x = symbols('x')

    # '^'를 '**'로 변경 및 다항식 정의
    ex = expression.replace("^", "**")
    polyn = sympify(ex)

    # 다항식 미분
    derivative = diff(polyn, x)
    return derivative

# 입력
polynomial_input = input("다항식 입력: ")

# 미분 계산
derivative_result = differentiate_polynomial(polynomial_input)

# 출력
print(f"도함수: {derivative_result}")

# Quiz2. n계 도함수 구하기
def calculate_nth_derivative(n):
    # 변수 및 함수 정의
    x = symbols('x')
    function = x * exp(x)

    # 도함수 계산
    nth_derivative = function
    for _ in range(n):
        nth_derivative = diff(nth_derivative, x)

    return nth_derivative

# n 입력
n = int(input("n값 입력: "))

# n번째 도함수 계산
result = calculate_nth_derivative(n)

# 출력
print(f"f의 {n}계 도함수: {result}")
