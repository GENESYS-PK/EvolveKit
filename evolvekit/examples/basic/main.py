from evolvekit.examples.basic.basic_example import basic_example
from evolvekit.examples.basic.basic_example_with_custom_evaluator import (
    basic_example_with_custom_evaluator,
)
from evolvekit.examples.basic.basic_example_with_simple_setup import (
    basic_example_with_simple_setup,
)


def main() -> None:
    """
    Run all basic examples in sequence.
    """
    print("EvolveKit – Basic Examples")
    print("=" * 60)

    print("\n1) The simplest example...")
    basic_example()

    print("\n2) The simple example with custom evaluator...")
    basic_example_with_custom_evaluator()

    print("\n3) The simple example with simple setup...")
    basic_example_with_simple_setup()

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
