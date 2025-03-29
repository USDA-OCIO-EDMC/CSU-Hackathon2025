
class Node:
    def __init__(self, x, y, ID):
        self.x = x
        self.y = y
        self.p = (x, y)
        self.ID = ID
        self.f = 0
        self.g = 0
        self.h = 0
        self.neighbors = []

    def to_string(self):
        return f'{self.ID}, {self.x}, {self.y}: {self.f} {self.g} {self.h}'

    def update_p(self):
        self.p = (self.x, self.y)
