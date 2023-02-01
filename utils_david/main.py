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

if __name__ == '__main__':
    # mails = get_data()
    # test_mail = mails.iloc[100]
    # my_flow = Flow("Enron_investigation")
    question1 = {"type": QuestionsType.YESNO, "question": 'Does the text suggest criminal activity', 'next_step': YesNoNextStep}
    question2 = {"type": QuestionsType.YESNO, "question": 'Does the enron stock mentioned?', 'next_step': YesNoNextStep}
    question3 = {"type": QuestionsType.YESNO, "question": 'Does the text mention hiding or destroying documents?', 'next_step': YesNoNextStep}
    question4 = {"type": QuestionsType.MAIL_ENTITIES, 'next_step': MailEntitiesNextStep}
    # questions = [question1, question2, question3]
    # add_question_func(questions)
    text = "Let's make sure that we don't get caught for this accounting fraud."
    chat = OpenAI(api_key=Flow.get_token_from_path(Path(ROOT)))
    res = chat.yes_or_no(text, "Does the text suggest criminal activity")