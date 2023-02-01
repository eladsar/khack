import pandas as pd

import openai
import json
import pathlib
import numpy as np
import Levenshtein as lev
from Levenshtein import ratio
from functools import partial

def get_edit_ratio(s1, s2):
    return lev.ratio(s1, s2)


def get_edit_distance(s1, s2):
    return lev.distance(s1, s2)


class OpenAI:

    def __init__(self, model="text-davinci-003", api_key=None, organization_id='org-7PVKrNOMnq1PvOkuHr1XJX88'):

        if api_key is None:

            path = pathlib.Path("openai_api_key.pkl")
            if path.exists():
                api_key = pd.read_pickle("openai_api_key.pkl")
            else:
                api_key = input("Please enter your OpenAI API key:")
                pd.to_pickle(api_key, "openai_api_key.pkl")

        self.model = model
        self.organization_id = organization_id
        self.api_key = api_key

        openai.api_key = api_key
        openai.organization = organization_id

    def file_list(self):
        return openai.File.list()

    def retrieve(self, model=None):
        if model is None:
            model = self.model
        return openai.Engine.retrieve(id=model)

    @staticmethod
    def models():
        return openai.Model.list()

    def embedding(self, text, model=None):
        if model is None:
            model = self.model
        response = openai.Engine(model).embedding(input=text, model=model)
        embedding = np.array(response.data[1]['embedding'])
        return embedding

    def ask(self, question, max_tokens=1024, temperature=1, top_p=1, frequency_penalty=0.0,
            presence_penalty=0.0, stop=None, n=1, stream=False, logprobs=None, echo=False):
        """
        Ask a question to the model
        :param n:
        :param logprobs:
        :param stream:
        :param echo:
        :param question:
        :param max_tokens:
        :param temperature: 0.0 - 1.0
        :param top_p:
        :param frequency_penalty:
        :param presence_penalty:
        :param stop:
        :return:
        """
        response = openai.Completion.create(
          engine=self.model,
          prompt=question,
          temperature=temperature,
          max_tokens=max_tokens,
          top_p=top_p,
          frequency_penalty=frequency_penalty,
          presence_penalty=presence_penalty,
          stop=stop,
          n=n,
          stream=stream,
          logprobs=logprobs,
          echo=echo
        )

        return response

    def summary(self, text, n_words=100, n_paragraphs=None, **kwargs):
        """
        Summarize a text
        :param text:  text to summarize
        :param n_words: number of words to summarize the text into
        :param n_paragraphs:   number of paragraphs to summarize the text into
        :param kwargs: additional arguments for the ask function
        :return: summary
        """
        if n_paragraphs is None:
            prompt = f"Task: summarize the following text into {n_words} words\nText: {text}\nResponse:"
        else:
            prompt = f"Task: summarize the following text into {n_paragraphs} paragraphs\nText: {text}\nResponse:"

        res = self.ask(prompt, **kwargs)
        return res.choices[0].text

    def question(self, text, question, **kwargs):
        """
        Answer a yes-no question
        :param text: text to answer the question from
        :param question: question to answer
        :param kwargs: additional arguments for the ask function
        :return: answer
        """
        prompt = f"Task: answer the following question\nText: {text}\nQuestion: {question}\nResponse:"

        res = self.ask(prompt, **kwargs)
        res = res.choices[0].text

        return res

    def yes_or_no(self, text, question, **kwargs):
        """
        Answer a yes or no question
        :param text: text to answer the question from
        :param question:  question to answer
        :param kwargs: additional arguments for the ask function
        :return: answer
        """
        prompt = f"Text: {text}\nTask: answer the following question with yes or no\nQuestion: {question}\nResponse:"

        res = self.ask(prompt, **kwargs)
        res = res.choices[0].text.lower().strip()
        if res == "yes":
            return True
        else:
            return False

    def classify(self, text, classes, **kwargs):
        """
        Classify a text
        :param text: text to classify
        :param classes: list of classes
        :param kwargs: additional arguments for the ask function
        :return: class
        """
        prompt = f"Task: classify the following text into one of the following classes\nText: {text}\nClasses: {classes}\nResponse:"

        res = self.ask(prompt, **kwargs)
        res = res.choices[0].text
        res = res.lower().strip()

        i = pd.Series(classes).str.lower().str.strip().apply(partial(get_edit_ratio, s2=res)).idxmax()

        return classes[i]

    def entities(self, text, humans=True, **kwargs):
        """
        Extract entities from a text
        :param humans:  if True, extract people, else extract all entities
        :param text: text to extract entities from
        :param kwargs: additional arguments for the ask function
        :return: entities
        """
        if humans:
            prompt = f"Task: extract people from the following text in a comma separated list\nText: {text}\nResponse:"
        else:
            prompt = f"Task: extract entities from the following text in a comma separated list\nText: {text}\nResponse:"

        res = self.ask(prompt, **kwargs)

        entities = res.choices[0].text.split(',')

        return entities

    def build_dataset(self, data=None, question=None, answer=None, path=None) -> object:
        """
        Build a dataset for training a model
        :param data: dataframe with prompt and completion columns
        :param question: list of questions
        :param answer: list of answers
        :param path: path to save the dataset
        :return: path to the dataset
        """
        if data is None:
            data = pd.DataFrame(data={'prompt': question, 'completion': answer})

        records = data.to_dict(orient='records')

        if path is None:
            print('No path provided, using default path: dataset.jsonl')
            path = 'dataset.jsonl'

        # Open a file for writing
        with open(path, 'w') as outfile:
            # Write each data item to the file as a separate line
            for item in records:
                json.dump(item, outfile)
                outfile.write('\n')

        return path
