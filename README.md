# khack
AI hackathon repo


# How to start

## Via Jupyter-lab
Connect to 192.168.10.45:46688 and navigate to hackathon_2023 directory
(located at /home/hackathon_2023).
Within the terminal, clone the git repository:

```bash
cd /home/hackathon_2023/ 
mkdir <your_name>
cd <your_name>

git clone https://github.com/eladsar/khack.git 
```

## Via pycharm

Clone the repository to your local machine and open the project in pycharm.
To run a remote interpreter, go to `File -> Settings -> Project -> Project Interpreter`
select `Add Remote` and fill in the following details:

```bash
Host: 192.169.10.45
Port: 46622
python interpreter path: /opt/conda/bin/python
```

## Work with OpenAI API

```python
from connect import OpenAI

model = OpenAI()

response = model.ask("Hello world")
```

## Train a model with the openai tool


