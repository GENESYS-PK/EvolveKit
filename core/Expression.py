from typing import List, Self
from Condition import Condition
from Concatenator import Concatenator


class Expression(Condition):
    """
    Represents a logical expression made up of conditions and concatenators.

    This class allows for the construction of complex logical expressions
    that can be evaluated as part of an evolution process.
    """

    def __init__(self):
        """
        Initialize the Expression class with an empty expression list.
        """
        super().__init__()
        self.expression_list: List[Condition|Concatenator] = []

    def evaluate(self, evolution: Self) -> bool:
        """
        Evaluate the logical expression based on the provided evolution instance.

        :param evolution: The evolution process instance to evaluate.
        :return: A boolean indicating the result of the logical expression.
        """
        if not self.expression_list:
            raise ValueError("The expression list is empty.")

        # Step 1: Evaluate all AND operations first
        and_results = []
        current_result = None
        current_concatenator = None

        for element in self.expression_list:
            if isinstance(element, Condition):
                condition_result = element.evaluate(evolution)

                if current_result is None:
                    current_result = condition_result
                elif current_concatenator == Concatenator.AND:
                    current_result = current_result and condition_result
                elif current_concatenator == Concatenator.OR:
                    # If OR encountered, store the current result and start a new AND evaluation
                    and_results.append(current_result)
                    current_result = condition_result

            elif isinstance(element, Concatenator):
                current_concatenator = element

        # Append the final result after last condition
        if current_result is not None:
            and_results.append(current_result)

        # Step 2: Evaluate the OR on the results of AND evaluations
        final_result = and_results[0]
        for res in and_results[1:]:
            final_result = final_result or res

        return final_result

    def begin(self, condition: Condition) -> Self:
        """
        Begin the logical expression with the first condition.

        :param condition: The initial condition or sub-expression to add.
        :return: The Expression instance, allowing for method chaining.
        :raises ValueError: If the expression has already been started.
        """
        if self.expression_list:
            raise ValueError("Expression already started with a condition.")

        self.expression_list.append(condition)
        return self

    def and_(self, condition: Condition) -> Self:
        """
        Add a condition to the expression with an AND operator.

        :param condition: The condition or sub-expression to add.
        :return: The Expression instance, allowing for method chaining.
        :raises ValueError: If no initial condition exists in the expression.
        """
        if not self.expression_list:
            raise ValueError("Expression must begin with a condition.")

        self.expression_list.append(Concatenator.AND)
        self.expression_list.append(condition)
        return self

    def or_(self, condition: Condition) -> Self:
        """
        Add a condition to the expression with an OR operator.

        :param condition: The condition or sub-expression to add.
        :return: The Expression instance, allowing for method chaining.
        :raises ValueError: If no initial condition exists in the expression.
        """
        if not self.expression_list:
            raise ValueError("Expression must begin with a condition.")

        self.expression_list.append(Concatenator.OR)
        self.expression_list.append(condition)
        return self
