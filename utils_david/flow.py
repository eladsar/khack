from connect import OpenAI
import pickle
from uuid import uuid4
from loguru import logger
import sys
from pathlib import Path
from questions import QuestionsType
from map_reducer import Map

logger.remove()
logger.add(sys.stdout, level='INFO', colorize=True,
           format='<green>{time:YYYY-MM-DD HH-mm-ss}</green> | <level>{level}</level> | <level>{message}</level>')

ROOT = '/home/hackathon_2023/'


class Flow:
    YesNo_COUNTER = 0
    Entity_COUNTER = 0
    Subject_COUNTER = 0
    OTHER_COUNTER = 0
    Mail_Entities_COUNTER = 0

    YesNo_RESPONSE = 0
    Entity_RESPONSE = 0
    Subject_RESPONSE = 0
    Other_RESPONSE = 0
    Mail_Entity_RESPONSE = 0

    def __init__(self, name, questions=None, next_step=None, open_ai: OpenAI = None):
        self.responses_path = Path(ROOT) / 'david' / 'response_logs' / name
        # check if a project with the same name exist
        if not self.responses_path.is_dir():
            self.responses_path.mkdir(parents=True, exist_ok=True)
        self.name = name  # id or nam of the flow
        if open_ai is None:
            self.client = OpenAI(api_key=self.get_token_from_path(Path(ROOT)))
        else:
            self.client = open_ai

        self.questions = questions  # list of questions (need to be of class type)
        self.count_questions()
        self.next_step = next_step  # list of responses of the questions
        self.count_next_step()
        self.id_tracer = []  # contain id for the questions and answer to retrieve data
        self.workflow = []
        self.workflow_traceback = []

    @staticmethod
    def get_token_from_path(p):
        try:
            with open(p / "openai_api_key.pkl", 'rb') as r:
                key = pickle.load(r)
            return key
        except Exception as ex:
            logger.info(f'Failed open key file, error: {ex}')
            return None

    def get_questions_count(self):
        return sum([self.YesNo_COUNTER, self.Subject_COUNTER, self.OTHER_COUNTER, self.Mail_Entities_COUNTER,
                    self.Entity_COUNTER])

    def get_next_step_count(self):
        return sum([self.Mail_Entity_RESPONSE, self.Subject_RESPONSE, self.Entity_RESPONSE, self.YesNo_RESPONSE,
                    self.Other_RESPONSE])

    def count_questions(self):
        for q in self.questions:
            if q['type'].name.lower() == "yesno":
                self.YesNo_COUNTER += 1
            elif q['type'].name.lower() == "entity":
                self.Entity_COUNTER += 1
            elif q['type'].name.lower() == "subject":
                self.Subject_COUNTER += 1
            elif q['type'].name.lower() == "mail_entities":
                self.Mail_Entities_COUNTER += 1
            else:
                self.OTHER_COUNTER += 1

    def count_next_step(self):
        for ns in self.questions:
            if 'yesno' in ns['next_step'].__name__.lower():
                self.YesNo_RESPONSE += 1
            elif 'entity' in ns['next_step'].__name__.lower():
                self.Entity_RESPONSE += 1
            elif 'subject' in ns['next_step'].__name__.lower():
                self.Subject_RESPONSE += 1
            elif 'mailentities' in ns['next_step'].__name__.lower():
                self.Mail_Entity_RESPONSE += 1
            else:
                self.Other_RESPONSE += 1

    @classmethod
    def from_questions(cls, name, questions):
        return cls(name=name, questions=questions)

    def build_flow(self):
        logger.info('building flow')
        # there must be a next step to a question ( can be None)
        if self.get_next_step_count() != self.get_questions_count():
            logger.info('You must have the same number of questions and next steps')
            raise Exception
        else:
            for i, task in enumerate(self.questions):
                tmp_id = uuid4()
                self.id_tracer.append(tmp_id)
                if task['type'] == QuestionsType.YESNO:
                    self.workflow.append({'type': task['type'], 'question': task['question'],
                                          'next_step': task['next_step'], 'id': tmp_id,
                                          "path": self.responses_path, 'client': self.client})
                else:
                    self.workflow.append({'type': task['type'], 'next_step': task['next_step'],
                                          'id': tmp_id, "path": self.responses_path, 'client': self.client})

        logger.info('finish building flow')

    def start(self, data):
        if len(self.workflow) > 0:
            logger.info('Starting flow')
            path_to_agg_answer = None
            for task in self.workflow:
                if task['type'] == QuestionsType.YESNO:
                    path_to_agg_answer = Map(data, task,path_to_agg_answer).map_reduce_yes_no_question()
                elif task['type'] == QuestionsType.MAIL_ENTITIES:
                    result = Map(data, task, path_to_agg_answer).map_reduce_mail_entities_question()
                else:
                    pass
        else:
            logger.error('No Flow defined')

    def add_question(self, question, type_=None):
        self.questions.append(question)
        self.count_questions()

    def add_next_step(self, next_step, type_=None):
        self.next_step.append(next_step)

