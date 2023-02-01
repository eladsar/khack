from connect import OpenAI
import pickle
from uuid import uuid4


ROOT = '/home/hackathon_2023/'
class Flow:
    YesNo_COUNTER = 0
    Entity_COUNTER = 0
    Subject_COUNTER = 0
    OTHER_COUNTER = 0

    YesNo_RESPONSE = 0
    Entity_RESPONSE = 0
    Subject_RESPONSE = 0
    Other_RESPONSE = 0

    def __init__(self, name, questions=None, next_step=None, open_ai:OpenAI =None):
        self.name = name  # id or nam of the flow
        self.id = uuid4()
        if open_ai is None:
            self.client = OpenAI(model='adda', api_key=self.get_token_from_path)
        else:
            self.client = open_ai

        self.questions = questions  # list of questions (need to be of class type)
        self.count_questions()
        self.count_next_step()
        self.next_step = next_step # list of responses of the questions

    @staticmethod
    def get_token_from_path(p):
        try:
            with open(p / "openai_api_key.pkl", 'rb') as r:
                key = pickle.load(r)
            return key
        except Exception as ex:
            print(f'Failed open key file, error: {ex}')
            return None

    def count_questions(self):
        for q in self.questions:
            if type(q).__name__ == "YesNo":
                self.YesNo_COUNTER += 1
            elif type(q).__name__ == "Entity":
                self.Entity_COUNTER += 1
            elif type(q).__name__ == "Subject":
                self.Subject_COUNTER += 1
            else:
                self.OTHER_COUNTER += 1

    def count_next_step(self):
        for ns in self.next_step:
            if type(ns).__name__== 'YesNo':
                self.YesNo_RESPONSE += 1
            elif type(ns).__name__ == 'Entity':
                self.Entity_RESPONSE += 1
            elif type(ns).__name__ == 'Subject':
                self.Subject_RESPONSE += 1
            else:
                self.Other_RESPONSE += 1

    @classmethod
    def from_questions(cls, name, questions):
        return cls(name=name, questions=questions)

    def build_flow(self):
        if len(self.questions) != len(self.next_step):
            raise Exception("You must have the same number of questions and next steps")
        else:
            for i, (q, ns) in enumerate(zip(self.questions, self.next_step)):
                self.client.ask(question=q)

    def add_question(self, question, type_=None):
        self.questions.append(question)
        self.count_questions()

    def add_next_step(self, next_step, type_=None):
        self.next_step.append(next_step)

