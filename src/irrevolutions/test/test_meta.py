# content of test_example.py
 
def add(x, y):
    return x + y

def test_addition():
    assert add(2, 3) == 5

def test_subtraction():
    assert add(5, -3) == 2
