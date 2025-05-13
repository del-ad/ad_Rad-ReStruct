class Question:

    def __init__(self, key, question, level, parent=None):
        self.key = key
        self.questions = [question]
        self.level = level
        self.children =[]
        self.parent = parent

    def is_question_present_in_children(self, question):
        if len(self.children) > 0:
            for child in self.children:
                if question in child.questions:
                    return True

    def __str__(self):
        return f'{self.key}: {self.questions}, level: {self.level}'
    def __repr__(self):
        return f'{self.key}: {self.questions}, level: {self.level}'