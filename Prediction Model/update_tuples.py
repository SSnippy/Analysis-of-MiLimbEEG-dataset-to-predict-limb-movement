import sys
import re

with open("generate_retrospective.py", "r", encoding="utf-8") as f:
    text = f.read()

def get_assignee(story_text):
    text_lower = story_text.lower()
    # model keywords
    if any(k in text_lower for k in ["svm", "csp", "eegnet", "cnn", "lda", "model", "train", "gridsearch", "selectkbest", "scaler", "normalize", "save x.npy"]):
        return '"Akshat Mittal"'
    else:
        return '"Agrim Pandey"'

def replacer(match):
    m_start = match.group(1)
    uid = match.group(2)
    story = match.group(3)
    status = match.group(4)
    prio = match.group(5)
    pts = match.group(6)
    end = match.group(7)
    
    assignee = get_assignee(story)
    
    return f'{m_start}{uid}, {story}, {status}, {prio}, {pts}, {assignee}{end}'

# Match exactly the 5-element user story tuples
pattern = r'( *\()("[^"]+"), ("[^"]+"), ("[^"]+"), ("[^"]+"), (\d+)(\),?)'
new_text = re.sub(pattern, replacer, text)

with open("generate_retrospective.py", "w", encoding="utf-8") as f:
    f.write(new_text)

print("Updated generate_retrospective.py safely.")
