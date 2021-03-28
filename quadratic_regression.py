from typing import Callable, Tuple, List
import random


Input = Tuple[float, float, float]
Coefficients = Tuple[float, float, float]
Gradient = Tuple[float, float, float]
Hypothesis = Callable[[Input], float]
Dataset = List[Tuple[Input, float]]


class QuadraticBasisPolynomials:
    def __init__(self):
        self.basis_functions = [
            lambda x: 1,
            lambda x: x[0],
            lambda x: x[1],
            lambda x: x[2],
            lambda x: x[0] * x[1],
            lambda x: x[0] * x[2],
            lambda x: x[1] * x[2],
            lambda x: x[0] * x[0],
            lambda x: x[1] * x[1],
            lambda x: x[2] * x[2],
        ]

    def __getitem__(self, index):
        return self.basis_functions[index]

    def __len__(self):
        return len(self.basis_functions)

    def linear_combination(self, weights: Coefficients) -> Hypothesis:
        def combined_function(x: Input) -> float:
            return sum(
                w * f(x)
                for (w, f) in zip(weights, self.basis_functions)
            )

        return combined_function


basis = QuadraticBasisPolynomials()


def total_error(weights: Coefficients, data: Dataset) -> float:
    hypothesis = basis.linear_combination(weights)
    return sum(
        (actual_output - hypothesis(example)) ** 2
        for (example, actual_output) in data
    )


def single_point_error(
        weights: Coefficients, point: Tuple[Input, float]) -> float:
    return point[1] - basis.linear_combination(weights)(point[0])


def gradient(weights: Coefficients, data_point: Tuple[Input, float]) -> Gradient:
    '''Return the gradient of the error with respect to each weight.'''
    error = single_point_error(weights, data_point)
    return [
            -2 * error * basis[i](data_point[0])
        for i in range(len(weights))
    ]


def print_debug_info(step, grad_norm, error, progress):
    print(f"{step}, {progress:.4f}, {error:.4f}, {grad_norm:.4f}")


def gradient_descent(
        data: Dataset,
        learning_rate: float,
        tolerance: float,
        training_callback = None,
) -> Hypothesis:
    weights = [
        random.random() * 2 - 1
        for i in range(len(basis))
    ]

    last_error = total_error(weights, data)
    step = 0
    progress = tolerance * 2
    grad_norm = 1

    if training_callback:
        training_callback(step, 0.0, last_error, 0.0)

    while abs(progress) > tolerance or grad_norm > tolerance:
        grad = gradient(weights, random.choice(data))
        grad_norm = sum(x**2 for x in grad)
        for i in range(len(weights)):
            weights[i] -= learning_rate * grad[i]

        error = total_error(weights, data)
        progress = error - last_error
        last_error = error
        step += 1

        if training_callback:
            training_callback(step, grad_norm, error, progress)

    return basis.linear_combination(weights)


def example_quadratic_data(num_points: int):
    def fn(x, y, z):
        return 2 - 4*x*y + z + z**2

    data = []
    for i in range(num_points):
        x, y, z = random.random(), random.random(), random.random()
        data.append(((x, y, z), fn(x, y, z)))

    return data


if __name__ == "__main__":
    data = example_quadratic_data(30)
    gradient_descent(
        data, 
        learning_rate=0.01, 
        tolerance=1e-06, 
        training_callback=print_debug_info
    )
