from connect import OpenAI
import pickle
from uuid import uuid4
from loguru import logger
import sys
from pathlib import Path
from questions import QuestionsType

logger.remove()
logger.add(sys.stdout, level='INFO', colorize=True,
           format='<green>{time:YYYY-MM-DD HH-mm-ss}</green> | <level>{level}</level> | <level>{message}</level>')

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

    def __init__(self, name, questions=None, next_step=None, open_ai: OpenAI = None):
        self.responses_path = Path(ROOT) / 'david' / 'response_logs' / name
        # check if a project with the same name exist
        if self.responses_path.is_dir():
            logger.error('This flow already exist')
            raise Exception
        else:
            # create the response directories
            self.responses_path.mkdir(parents=True, exist_ok=False)
            self.name = name  # id or nam of the flow
            if open_ai is None:
                self.client = OpenAI(model='adda', api_key=self.get_token_from_path)
            else:
                self.client = open_ai

            self.questions = questions  # list of questions (need to be of class type)
            self.count_questions()
            self.count_next_step()
            self.next_step = next_step  # list of responses of the questions
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
            if type(ns).__name__ == 'YesNo':
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
        logger.info('building flow')
        # there must be a next step to a question ( can be None)
        if len(self.questions) != len(self.next_step):
            logger.info('You must have the same number of questions and next steps')
            raise Exception
        else:
            for i, (q, ns) in enumerate(zip(self.questions, self.next_step)):
                tmp_id = uuid4()
                self.id_tracer.append(tmp_id)
                self.workflow.append({'question': q, 'next_step': ns, 'id': tmp_id})


        logger.info('finish building flow')

    def start(self, data):
        if len(self.workflow) > 0:
            logger.info('Starting flow')
            for task in self.workflow:
                # res = self.client.ask(task['question'])
                # self.write_response(self.responses_path / task['id'])
                # is_next_step, data_for_next_step = task['next_step'].next()
                # if not is_next_step:
                #     break
                if task['question'] == QuestionsType.YESNO:
                    pass
                elif task['question'] == QuestionsType.ENTITIES:
                    pass
                else:
                    pass
        else:
            logger.error('No Flow defined')

    def add_question(self, question, type_=None):
        self.questions.append(question)
        self.count_questions()

    def add_next_step(self, next_step, type_=None):
        self.next_step.append(next_step)

    @staticmethod
    def write_response(path):
        res = None
        try:
            with open(path, 'r') as reader:
                res = reader.read()
        except Exception as ex:
            logger.error(f'failed to write response, error {ex}')
        return res

    def ask_yes_no_question(self):
        pass