from evolvekit.examples.advanced.optimize_sphere import optimize_sphere
from evolvekit.examples.advanced.optimize_rastrigin import optimize_rastrigin
from evolvekit.examples.advanced.optimize_rosenbrock import optimize_rosenbrock
from evolvekit.examples.advanced.compare_clamp_strategies import (
    compare_clamp_strategies,
)


def main() -> None:
    """
    Run all advanced examples in sequence.
    """
    print("EvolveKit – Advanced Examples")
    print("=" * 60)

    print("\n1) Optimizing Sphere...")
    optimize_sphere()

    print("\n2) Optimizing Rastrigin...")
    optimize_rastrigin()

    print("\n3) Optimizing Rosenbrock...")
    optimize_rosenbrock()

    print("\n4) Comparing clamp strategies...")
    compare_clamp_strategies()

    print("\n" + "=" * 60)
    print("Done. Check the generated CSV files for details.")


if __name__ == "__main__":
    main()
