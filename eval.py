import pandas as pd
import re
import sys

# usage: python eval.py [responses.csv] [answers.csv]
response_path = sys.argv[1] if len(sys.argv) > 1 else 'responses.csv'
answer_path = sys.argv[2] if len(sys.argv) > 2 else 'samples.csv'

def extract_answer(response):
    pattern = r"\[ANSWER\]:\s*\(([A-J])\)"
    match = re.search(pattern, response)
    if match:
        return '('+match.group(1)+')'
    else:
        print("Answer extraction failed:\n" + response)
        return None


res = pd.read_csv(response_path)
responses = res['responses']
ans = pd.read_csv(answer_path)
answers = ans['answers']

correct = 0
for idx, response in enumerate(responses):
    std_ans = extract_answer(response)
    if std_ans:
        if str(answers[idx]).strip() == str(std_ans):
            correct += 1

print(f"acc: {(correct/len(answers))*100}%")
