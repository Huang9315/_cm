# =========================================================
# Finite Field Homework
# GF(p) implementation with group & field axioms
# =========================================================

import random
from abc import ABC, abstractmethod

# =========================================================
# Abstract Group Definition
# =========================================================

class Group(ABC):
    """Abstract base class for mathematical groups"""

    @property
    @abstractmethod
    def identity(self):
        pass

    @abstractmethod
    def operation(self, a, b):
        pass

    @abstractmethod
    def inverse(self, a):
        pass

    @abstractmethod
    def include(self, element):
        pass

    def random_generate(self):
        elements = self._get_all_elements()
        return random.choice(elements)

    def _get_all_elements(self):
        raise NotImplementedError


# =========================================================
# Finite Field Element
# =========================================================

class FiniteFieldElement:
    """Element of finite field GF(p)"""

    def __init__(self, value, prime):
        self.prime = prime
        self.value = value % prime

    def __eq__(self, other):
        return (
            isinstance(other, FiniteFieldElement) and
            self.value == other.value and
            self.prime == other.prime
        )

    def __repr__(self):
        return f"GF({self.prime})({self.value})"


# =========================================================
# Additive Group of GF(p)
# =========================================================

class FiniteFieldAddGroup(Group):
    """Additive group of finite field GF(p)"""

    def __init__(self, prime):
        self.prime = prime
        self._identity = FiniteFieldElement(0, prime)

    @property
    def identity(self):
        return self._identity

    def operation(self, a, b):
        return FiniteFieldElement(a.value + b.value, self.prime)

    def inverse(self, a):
        return FiniteFieldElement(-a.value, self.prime)

    def include(self, element):
        return isinstance(element, FiniteFieldElement) and element.prime == self.prime

    def _get_all_elements(self):
        return [FiniteFieldElement(i, self.prime) for i in range(self.prime)]


# =========================================================
# Multiplicative Group of GF(p) (excluding 0)
# =========================================================

class FiniteFieldMulGroup(Group):
    """Multiplicative group of finite field GF(p)"""

    def __init__(self, prime):
        self.prime = prime
        self._identity = FiniteFieldElement(1, prime)

    @property
    def identity(self):
        return self._identity

    def operation(self, a, b):
        return FiniteFieldElement(a.value * b.value, self.prime)

    def inverse(self, a):
        if a.value == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse")
        # Fermat's little theorem
        inv = pow(a.value, self.prime - 2, self.prime)
        return FiniteFieldElement(inv, self.prime)

    def include(self, element):
        return (
            isinstance(element, FiniteFieldElement) and
            element.prime == self.prime and
            element.value != 0
        )

    def _get_all_elements(self):
        return [FiniteFieldElement(i, self.prime) for i in range(1, self.prime)]


# =========================================================
# Finite Field GF(p)
# =========================================================

class FiniteField:
    """Finite field GF(p)"""

    def __init__(self, prime):
        self.prime = prime
        self.add_group = FiniteFieldAddGroup(prime)
        self.mul_group = FiniteFieldMulGroup(prime)

    def element(self, value):
        return FiniteFieldElement(value, self.prime)


# =========================================================
# Operator Overloading Version
# =========================================================

class GFNumber:
    """Finite field number with operator overloading"""

    def __init__(self, field, value):
        self.field = field
        self.elem = field.element(value)

    def __add__(self, other):
        r = self.field.add_group.operation(self.elem, other.elem)
        return GFNumber(self.field, r.value)

    def __sub__(self, other):
        inv = self.field.add_group.inverse(other.elem)
        r = self.field.add_group.operation(self.elem, inv)
        return GFNumber(self.field, r.value)

    def __mul__(self, other):
        r = self.field.mul_group.operation(self.elem, other.elem)
        return GFNumber(self.field, r.value)

    def __truediv__(self, other):
        inv = self.field.mul_group.inverse(other.elem)
        r = self.field.mul_group.operation(self.elem, inv)
        return GFNumber(self.field, r.value)

    def __repr__(self):
        return f"GF({self.field.prime})({self.elem.value})"


# =========================================================
# Field Axiom Checks
# =========================================================

def check_distributivity(field, trials=50):
    for _ in range(trials):
        a = field.mul_group.random_generate()
        b = field.add_group.random_generate()
        c = field.add_group.random_generate()

        left = field.mul_group.operation(
            a,
            field.add_group.operation(b, c)
        )
        right = field.add_group.operation(
            field.mul_group.operation(a, b),
            field.mul_group.operation(a, c)
        )

        assert left == right


def check_field(field):
    check_distributivity(field)


# =========================================================
# Demo
# =========================================================

if __name__ == "__main__":
    F = FiniteField(7)

    # basic operations
    a = F.element(3)
    b = F.element(5)

    print("Add:", F.add_group.operation(a, b))
    print("Mul:", F.mul_group.operation(a, b))
    print("Inv:", F.mul_group.inverse(b))

    # operator overloading demo
    x = GFNumber(F, 3)
    y = GFNumber(F, 5)

    print("x + y =", x + y)
    print("x * y =", x * y)
    print("x / y =", x / y)

    # axiom check
    check_field(F)
    print("All field axioms passed.")
