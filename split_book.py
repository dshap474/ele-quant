import os
import re

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

with open('book', 'r') as f:
    lines = f.readlines()

# Identify start indices and titles
indices = []
titles = []
for i, line in enumerate(lines):
    if line.startswith('**Introduction**'):
        indices.append(i)
        titles.append('Introduction')
    elif line.startswith('**Notation**'):
        indices.append(i)
        titles.append('Notation')
    elif re.match(r'\*\*(?:[Cc]hapter|CHAPTER)', line):
        match = re.match(r'\*\*(?:[Cc]hapter|CHAPTER)\s*(\d+)', line)
        if match:
            indices.append(i)
            titles.append(f'Chapter {match.group(1)}')
    elif line.startswith('**References**'):
        indices.append(i)
        titles.append('References')
    elif line.startswith('**Index**'):
        indices.append(i)
        titles.append('Index')

indices.append(len(lines))

os.makedirs('markdown', exist_ok=True)

for j in range(len(indices)-1):
    start = indices[j]
    end = indices[j+1]
    title = titles[j]
    filename = f'{j:02d}-{slugify(title)}.md'
    with open(os.path.join('markdown', filename), 'w') as out:
        out.writelines(lines[start:end])
