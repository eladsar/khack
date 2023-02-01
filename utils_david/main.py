from flow import Flow
from pathlib import Path
import pandas as pd
from questions import QuestionsType, get_question_func
from questions import YesNoNextStep, MailEntitiesNextStep
from connect import OpenAI
ROOT = '/home/hackathon_2023/'

test_text = "Immediately on landing in Jaffa, 7 September 1906, Ben Gurion set off,"
" on foot, in a group of fourteen, to Petah Tikva.[19][20] "
"It was the largest of the 13 Jewish agricultural settlements and consisted of 80 households"
" with a population of nearly 1500;"
" of these around 200 were Second Aliyah pioneers like Ben Gurion."
" He found work as a day labourer, waiting each morning hoping to be chosen by an overseer."
" Jewish workers found it difficult competing with local villagers who were more skilled"
" and prepared to work for less. Ben Gurion was shocked at the number of Arabs employed."
" In November he caught malaria and the doctor advised he return to Europe."
" By the time he left Petah Tikva in summer of 1907 he had worked an average 10 days a month"
" which frequently left him with no money for food.[21][22]"
" He wrote long letters in Hebrew to his father and friends. "
"They rarely revealed how difficult life was. Others who had come from Płońsk"
" were writing about tuberculosis, cholera and people dying of hunger.[23]"


def get_data(path=None, limit=-1):
    if not path:
        path = Path(ROOT) / 'data' / 'enron' / 'emails.parquet'
    df = pd.read_parquet(path=path, engine='pyarrow')
    return df

def add_question_func(list_of_questions):
    for q in list_of_questions:
        q['question_func'] = get_question_func(q=q)


def fake_dataset(df):
    qs1 = ['Can you change the accounting treatment on this transaction? We need to make the earnings look better.',
          "Let's make sure that these transactions don't appear on the books.",
          "Can we make this transaction look more favorable on the books?",
          "Let's make sure that we don't get caught for this accounting fraud.",
          ]
    qs2 = ["I think you will be interested in the stock movement of Enron.",
           "We need to sell all of our Enron stock before it's too late.",
           "I just found out some confidential information about Enron's financials."
           " Let's sell our stock before anyone else finds out.",
           "I just heard some rumors that the stock is going to drop soon."]

    qs3 = ["Can you help me destroy this email? I don't want anyone to see it.",
           "Let's make sure that this information doesn't get out. Can you help me keep it confidential?",
           "Can you delete these emails? I don't want them to be seen by anyone.",
           "Let's make sure that we don't leave any evidence behind. Can you help me destroy this information?"]

    for i in range(len(qs1)):
        df.at[i, "body"] = qs1[i]

    for i in range(5, 5+len(qs2)):
        df.at[i, "body"] = qs2[i-5]

    for i in range(10, 10+len(qs3)):
        df.at[i, "body"] = qs3[i-10]


if __name__ == '__main__':
    question1 = {"type": QuestionsType.YESNO, "question": 'Does the text suggest criminal activity?', 'next_step': YesNoNextStep}
    question2 = {"type": QuestionsType.YESNO, "question": 'Does the enron stock mentioned?', 'next_step': YesNoNextStep}
    question3 = {"type": QuestionsType.YESNO, "question": 'Does the text mention hiding or destroying documents?', 'next_step': YesNoNextStep}
    question4 = {"type": QuestionsType.MAIL_ENTITIES, 'next_step': MailEntitiesNextStep}
    questions = [question1, question2, question3, question4]
    add_question_func(questions)
    # mails = get_data()
    # raise Exception
    test_mail = pd.read_csv('test_mail.csv')[0:20]
    fake_dataset(test_mail)
    my_flow = Flow.from_questions("enron_investigation", questions)
    my_flow.build_flow()
    my_flow.start(test_mail)
