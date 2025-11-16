from enum import Enum, auto


class GaClampStrategy(Enum):
    """
    The purpose of this enumerator is to define strategies for restoring gene values to the domain in
    real-valued chromosomes.

    Assume the gene value is 3 and the domain is [-8, 1] for the examples below.

    :cvar NONE: Clamp strategy is disabled (in the example: 3)
    :cvar CLAMP: The value should be clamped to the range (in the example: 1)
    :cvar BOUNCE: The value should "bounce" off the boundary by the amount the gene exceeded the allowed range (in the example: -1)
    :cvar OVERFLOW: The value should "wrap around" (in the example: -6)
    :cvar RANDOM: Assign a random value within the range (in the example: any value from the range [-8, 1])

    For the BOUNCE and OVERFLOW algorithms, if a gene appears that is significantly outside the problem domain,
    the "bounce" or "wrap-around" should occur repeatedly.

    Example:

    -   BOUNCE: The gene value is 16, so we have exceeded the domain by 15. We bounce "to the left" and get gene -8.
        However, the remainder is 15-(1-(-8)) = 6, so we bounce again, this time "to the right". We get -2, because -8+6 = -2.

    -   OVERFLOW: Analogous to BOUNCE, but instead of bouncing off the boundaries, the value wraps around
        just like integer overflow in computer systems.
    """

    NONE = auto()
    CLAMP = auto()
    BOUNCE = auto()
    OVERFLOW = auto()
    RANDOM = auto()
