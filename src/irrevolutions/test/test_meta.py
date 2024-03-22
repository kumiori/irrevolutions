# We meta-test. We test the test. We test the tester.
# The content of it all is a story that's told by the Mole.
# How many times has the Mole seen, as a marketing trick,
# bullshit like: "1+1=3" or "12+6=36", the latter on the packaging
# of a roll of toilet paper seen in an Auchan supermarket, in Paris
# no later than a few days ago. A link to a shot will appear, a
# a future commit(ment) trick. However this may seem trivial,
# testing is a way to perturb the flimsy roots of an unstable system.
# With a proof.

def add(x, y):
    return x + y


def test_addition():
    assert add(1, 1) == 2


def test_subtraction():
    assert add(5, -3) == 2
