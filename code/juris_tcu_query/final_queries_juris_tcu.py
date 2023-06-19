status = 'id'
statements = []
questions = []
expressions = []
qtty_statements = 0
total_statements = 0
for line in open('data/llm_juris_tcu/query_llm_selecionada.txt', encoding='Utf-8'):
    if line.startswith('-----') and status in ['line', 'expression']:
        qtty_statements += 1
        if qtty_statements == len(statements):
            assert (qtty_statements == 1 or qtty_statements == line.count('='))
            #print('statements:', statements)
            #if status == 'line':
            #    print('queries:', questions[-1], '/', expressions[-1])
            #elif status == 'expression':
            #    print('queries:', questions[-1], '/ (idem)')
            total_statements += qtty_statements
            #print(line[:-1])
            statements = []
            qtty_statements = 0
    elif line.startswith('#'):
        id = int(line[1:-1])
        status = 'statement'
    elif status == 'statement':
        status = 'question'
        statements.append((id, line[:-1]))
    elif status == 'question':
        status = 'expression'
        questions.append(line[:-1])
    elif status == 'expression':
        status = 'line'
        expressions.append(line[:-1])
    elif line.startswith('=====') and status in ['line', 'expression']:
        break
    else:
        print('?', line)
        assert(False)
print('questions:', len(questions), 'expressions:', len(expressions), 'statements:', total_statements)
with open('data/juris_tcu/query2.txt', 'w') as f_out:
    for question in questions[:50]:
        f_out.write(question + '\n')
with open('data/juris_tcu/query3.txt', 'w') as f_out:
    for expression in expressions[:50]:
        f_out.write(expression + '\n')
        
    