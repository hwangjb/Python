class Person:
    count = 0

    def __init__(self, name):
        self.name = name
        Person.count += 1
        print(self.name + "is initialized")

    def work(self, company):
        print(self.name + "is working in" + company)

    def sleep(self):
        print(self.name + "is slepping")

    @classmethod
    def getCount(cls):
        return cls.count


