class A(object):
    def __call__(*args):
        print args

def foo(*args):
    print args

class B(object):
    
    def __init__(self):
        self.A = A()
    foo = foo
    def bar(*args):
        print args
