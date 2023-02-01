import pandas as pd

import openai
import json
import pathlib
import numpy as np


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
