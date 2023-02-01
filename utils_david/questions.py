from connect import OpenAI
from enum import Enum


def get_question_func(q):
    if type(q).__name__ == 'dict':
        q = q['type']

    if q == QuestionsType.YESNO:
        return OpenAI.yes_or_no
    elif q == QuestionsType.ENTITIES:
        return OpenAI.entities
    elif q == QuestionsType.TITLE:
        return OpenAI.title
    elif q == QuestionsType.SUMMARY:
        return OpenAI.summary
    elif q == QuestionsType.CLASSIFY:
        return OpenAI.classify
    elif q == QuestionsType.MAIL_ENTITIES:
        return OpenAI.mail_entities

class QuestionsType(Enum):
    YESNO = 1
    ENTITIES = 2
    TITLE = 3
    SUMMARY = 4
    CLASSIFY = 5
    MAIL_ENTITIES = 6


class YesNoNextStep:
    # if yes continue to next question else do nothing
    def __init__(self, res_id):
        with open(res_id) as reader:
            res = reader.read()
        self.res = res

    def next(self):
        if self.res.lower() in ['yes', 'yeah']:
            return True, None
        return None, None


class MailEntitiesNextStep:
    def __init__(self, res_id):
        with open(res_id) as reader:
            res = reader.read()
        self.res = res

    def next(self):
        if self.res:
            return True, self.res
        return None, None


class SummaryNextStep:
    pass

class TitleNextStep:
    pass

class ClassifyNextStep:
    pass


class YesNoBlock:

    def __init__(self, question, ns_question):
        self.question = question
        self.next_step = ns_question

