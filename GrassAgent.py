import random as rnd


class Grass:

    ID = -1
    ptype = 0

    def __init__(self, x, y, reproductionRate, consumeRate):
        self.x = x
        self.y = y
        self.reproductionRate = reproductionRate
        self.food = 1
        self.consumeRate = consumeRate

    def update(self):
        r = rnd.uniform(0, 1)
        offspring = 0
        if r < self.reproductionRate:
            offspring = Grass(self.x, self.y, self.reproductionRate, self.consumeRate)
        return offspring

    def consume(self):
        self.food -= self.consumeRate
        if self.food <= 0:
            return 0
        return 1
