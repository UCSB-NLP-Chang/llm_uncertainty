def load_system_prompt(dataset_name):
    if dataset_name == 'ambigqa':
        return ''
    elif dataset_name == "ambig_inst":
        return "Think step by step to finish the following task. At the end of your answer, you need to format the final solution in a specific way (refer to the task for more details about the format)."
    elif dataset_name == "nq_open":
        return ''
    elif dataset_name == "gsm8k":
        return "Follow the given examples and answer the question."

def load_fewshot_prompt(dataset_name):
    if dataset_name == 'ambigqa':
        with open("lib_prompt/forward/ambigqa_fewshot.txt", 'r',encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt
    elif dataset_name == "ambig_inst":
        return ''
    elif dataset_name == "nq_open":
        with open("lib_prompt/forward/nq_fewshot.txt", 'r',encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt
    elif dataset_name == "gsm8k":
        with open("lib_prompt/forward/gsm8k_fewshot.txt", 'r',encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt


def inst_transform(orig_inst, clarified_inst):
    if "Rearrange the objects" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {object name 1}, {object name 2}, ..., {object name n}" (Only output the name of the rearranged objects).'''
    elif "Sort the data in alphabetical order" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {data 1}, {data 2}, ..., {data n}. '''
    elif "Organize the files by date" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {file name 1}, {file name 2}, ..., {file name n}" (Only output the name of the rearranged files without other attributes. Include the extension name if it is provided).'''
    elif "Calculate the average of the numbers in the given" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the average number}" (only output the number).'''
    elif "Find the middle value in a list of numbers" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the middle number}" (only output the number).'''
    elif "Determine the length of a sentence" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the length of the sentence}" (Write only the number, without units).'''
    elif "Sort the names alphabetically" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {name 1}, {name 2}, ..., {name n}" (Write only the name).'''
    elif "Determine the square root of a number" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is "{the square root number(s)}" (write multiple numbers if necessary).'''
    elif "Identify the subject in the sentence" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is "{the subject}" (either a word or a sentence depending on your output).'''
    elif "Find the capital of a country" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is "{the capital}" (only output the city name).'''
    elif "Classify a movie based on its rating" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is "{the category you classify}" (only output the category).'''
    elif "Select the longest sentence from the following choices, output the sentence index" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is "{sentence index}" (only output the index).'''
    elif "Count the number of objects in the given" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is "{number of objects}" (only output the number).'''
    elif "Rank the football players based on their performance" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {player name 1}, {player name 2}, ..., {player name n}" (Only output the name of the rearranged files without other statistics).'''
    elif "Identify the largest city in the set" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {city name}" (Only output the city name without other attributes).'''
    elif "Write the inputted word with a space between each letter in lowercase" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the letter list}".'''
    elif "Write the first letter of the input word in lowercase." in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the first letter}".'''
    elif "Write the second letter of the input word in lowercase." in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the second letter}".'''
    elif "Identify the first word, reading from left to right, in the input sentence" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the word you find}".'''
    elif "Pluralize the input English word" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the plural word you find}".'''
    elif "Identify the larger animal in the input" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the animal}".'''
    elif "Add the two numerical inputs together" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the result of addition (a number)}".'''
    elif "Subtract the second number from the first number" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the result of subtraction (a number)}".'''
    elif "Convert this positive integer number (in decimal" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {a word}".'''
    elif "Write the antonym of the given word. The" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the antonym}".'''
    elif "Write the antonym of the given word. The" in orig_inst:
        clarified_inst = clarified_inst + ''' At the end of your output, highlight the final solution using the format: "The answer is {the antonym}".'''
    return clarified_inst

example_clarify = '''Original Question: {orig_question}
Clarifications:
{all_clarifications}'''

clarify_format = '''{idx}. {clarification}'''


def load_clarification_system_prompt(dataset_name):
    if dataset_name == 'ambigqa':
        with open('lib_prompt/clarification/ambigqa_system.txt','r',encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt
    elif dataset_name == "ambig_inst":
        return ""
    elif dataset_name == "nq_open":
        with open('lib_prompt/clarification/nqopen_system.txt','r',encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt
    elif dataset_name == "gsm8k":
        with open('lib_prompt/clarification/gsm8k_system.txt','r',encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

def load_clarification_user_prompt(dataset_name):
    if dataset_name == 'ambigqa':
        return ''
    elif dataset_name == "ambig_inst":
        with open('lib_prompt/clarification/ambiginst_user.txt','r',encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt
    elif dataset_name == "nq_open":
        return ''
    elif dataset_name == "gsm8k":
        return ""


