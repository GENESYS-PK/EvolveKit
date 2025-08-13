import numpy as np

from evolvekit.core.benchmarks.SphereEvaluator import SphereEvaluator
from evolvekit.core.benchmarks.RastriginEvaluator import RastriginEvaluator
from evolvekit.core.benchmarks.RosenbrockEvaluator import RosenbrockEvaluator
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs


def make_args(vec):
    """
    Create GaEvaluatorArgs from a real-valued vector.

    :param vec: one-dimensional sequence of floats (real chromosome)
    :returns: GaEvaluatorArgs wrapping a GaIndividual with given real_chrom
    :raises: None
    """
    ind = GaIndividual(real_chrom=np.array(vec, dtype=np.float64))
    return GaEvaluatorArgs(ind)


def show_assert_equal(label, got, expected):
    """
    Print a formatted assertion message and check equality.

    :param label: Human-readable label for the assertion.
    :param got: The actual value obtained.
    :param expected: The expected value to compare against.
    :raises AssertionError: If the values are not equal.
    :return: None
    """
    print(f"[assert] {label}: expected={expected}, got={got}")
    assert got == expected, f"{label} expected {expected}, got {got}"


def test_sphere_zero_is_zero():
    """
    For the Sphere function, the value at the zero vector equals 0.

    :returns: None
    :raises: None
    """
    ev = SphereEvaluator(dim=5)
    args = make_args([0, 0, 0, 0, 0])
    val = ev.evaluate(args)
    dom = ev.real_domain()
    print(f"\nSphere.evaluate -> {val}; domain len={len(dom)}, sample={dom[0] if dom else None}")
    show_assert_equal("Sphere f(0)", val, 0.0)
    show_assert_equal("Sphere domain length", len(dom), 5)


def test_rastrigin_zero_is_zero():
    """
    For the Rastrigin function, the value at the zero vector equals 0.

    :returns: None
    :raises: None
    """
    ev = RastriginEvaluator(dim=3)
    args = make_args([0, 0, 0])
    val = ev.evaluate(args)
    dom = ev.real_domain()
    print(f"\nRastrigin.evaluate -> {val}; domain len={len(dom)}, sample={dom[0] if dom else None}")
    show_assert_equal("Rastrigin f(0)", val, 0.0)
    show_assert_equal("Rastrigin domain length", len(dom), 3)


def test_rosenbrock_properties():
    """
    For the Rosenbrock function:
    - at [0, 0, 0], the value equals n-1 (=2 for n=3),
    - at the global minimum [1, 1, 1], the value equals 0.

    :returns: None
    :raises: None
    """
    ev = RosenbrockEvaluator(dim=3)
    # At zeros: value = (n-1) because (1 - x_i)^2 = 1 for i=0..n-2; the other term is 0
    args_zero = make_args([0, 0, 0])
    val_zero = ev.evaluate(args_zero)
    print(f"\nRosenbrock f([0,0,0]) -> {val_zero}")
    show_assert_equal("Rosenbrock f([0,0,0])", val_zero, 2.0)

    # Global minimum at ones vector -> 0
    args_one = make_args([1, 1, 1])
    val_one = ev.evaluate(args_one)
    print(f"Rosenbrock f([1,1,1]) -> {val_one}")
    show_assert_equal("Rosenbrock f([1,1,1])", val_one, 0.0)

    dom = ev.real_domain()
    print(f"Rosenbrock domain len={len(dom)}, sample={dom[0] if dom else None}")
    show_assert_equal("Rosenbrock domain length", len(dom), 3)