import json
import random
import copy
import csv
from io import StringIO
from contextlib import redirect_stdout
from tqdm import tqdm
import os
import re
# import regex as re
import math
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from scripts.generate_gsm8k_programs import * 

import warnings
warnings.filterwarnings("ignore")

base_path   = "math_data"
common_path = ""
common_save_path = "_{}_".format(sys.argv[1])
executed_programs_path = "executed_programs.json"
new_params_path = 'train_new_parameters.json'
new_ques_ans_path = 'questions_answer.json'
new_ques_path = 'train_new_questions.json'

DEBUG_MODE = False

def debug_print(*args, **kwargs):
    """Custom print function that prints only when DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

torch.random.manual_seed(0)


program_data = []

PROMPT_DICT = {
        "prompt_new_parameter_value": (
            "Here is a math question with the parameter and parameter values. Please perturb the value of parameters into different values. Output five kinds of new values in the same format as the given parameters in five lines without index.\n\n"
            "Question:\n{question}\n\nParameters:\n{parameters}\n\n"
        ),
        "prompt_rewrite_question": (
            "Here is a math question with old parameter values, and five kinds of new parameter values. Please rewrite the question five times to update all the parameters from old value to each corresponding new value in five lines without index.\n\n"
            "Question: :\n{question}\n\nOld Parameters:\n{parameters}\n\nNew Parameters:\n{new_parameters}\n\nNew Question:"
        ),
        "prompt_answer_question": (
            "Answer the math question below. Only output the answer without units and any context words.\n\n"
            "Question:\n{question}\n\nAnswer:"
        ),
        "prompt_answer_question_few_shot_cot": (
            "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
            "A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\n"
            "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
            "A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\n"
            "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
            "A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\n"
            "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
            "A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\n"
            "Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n"
            "A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\n"
            "Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n"
            "A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\n"
            "Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n"
            "A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\n"
            "Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n"
            "A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n"
            "Q: {question}\n"
            "A: "
        )
    }

def call_function_with_args(parameter):
    outstr = '\noutput = answer({})\nprint(output)'.format(parameter)
    return outstr

def get_last_number(text):
    # Regular expression to match numbers with optional commas and decimals
    pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'

    # Find all matches
    matches = re.findall(pattern, text)

    # Get the last match if it exists
    last_number = matches[-1] if matches else "-50000000000"

    # Remove commas and convert to float if a match is found
    if last_number:
        last_number = float(last_number.replace(',', ''))
    return last_number

def get_perturb_parameter_value_prompt_messages(case):
    debug_print("Inside perturb_parameter_value function .... ")
    prompt_template = PROMPT_DICT['prompt_new_parameter_value']
    prompt = prompt_template.format_map(
        {
            "question": case['question'], 
            "parameters": case['parameters']
        })

    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    debug_print("Prompt : ", prompt)
    return messages

def clear_parameter_output(response):
    response = response.replace('Parameters:', '').replace('```', '').replace('- ', '').strip().split('\n')
    for idx, para in enumerate(response):
        para.replace(str(idx)+'. ', '').strip()
        para.replace(str(idx+1) + '. ', '').strip()
    response = [x for x in response if len(x) > 0]
    return response

def find_numbers(text):
    # Regular expression to match numbers with optional commas and decimals
    pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'

    # Find all matches
    matches = re.findall(pattern, text)

    # Get the last match if it exists
    last_number = matches[-1] if matches else None

    # Remove commas and convert to float if a match is found
    if last_number:
        last_number = float(last_number.replace(',', ''))
    return last_number


import random
xx = random.randint(1, 1000)

def execute(program):
    f_open = open("/tmp/.execution_gen_file_math_x{}.py".format(xx), "w")
    f_open.flush()
    f_open.write(program)
    f_open.flush()
    f_open.close()
    os.system("timeout 30 stdbuf -oL python -W ignore /tmp/.execution_gen_file_math_x{}.py > /tmp/.execution_gen_file_math_output_x{}.txt; cat /tmp/.execution_gen_file_math_output_x{}.txt".format(xx, xx, xx))
    result = open("/tmp/.execution_gen_file_math_output_x{}.txt".format(xx)).read().strip()
    return result

def generate_new_parameter_value_parallel(pth=None):
    with open(base_path + common_path + executed_programs_path, 'r') as infile:
        program_data = json.load(infile)

    final_pth = base_path + common_path + common_save_path + new_ques_ans_path
    if os.path.exists(final_pth):
        with open(final_pth, 'r') as infile:
            program_data = json.load(infile)

    correct_counter = 0
    compile_fail_counter = 0

    correct_case = []

    for idx, case in enumerate(program_data):
        if idx in [306]:
            continue
        # renamed selected_programs to candidate_programs
        if len(case['candidate_programs']) == 0:
            continue

        # program = case['candidate_programs'][0]
        for program in case['candidate_programs']:
            if 'math.' in program:
                program = 'import math\n\n' + program
            program_copy = copy.deepcopy(program)
            program += call_function_with_args(case['parameters'])
            try:
                # f = StringIO()
                # with redirect_stdout(f):
                #     exec(program)
                s = execute(program) # f.getvalue().strip()
                debug_print("---- Program", program)
                debug_print("---- Program_output", s)
                debug_print("---- Case_answer", case['answer'])
                debug_print("---- Case_answer_processed", case['answer'].split()[-1])
                debug_print("\n\n" + "-" * 30 + "\n\n")
                # if round(float(s)) == round(float(case['answer'])):
                if round(float(s)) == round(float(case['answer'].split()[-1])):
                    case['candidate_programs'][0] = program_copy
                    correct_counter += 1
                    correct_case.append(case)
                    break
            except Exception as e:
                print(e)
                print(case)
                compile_fail_counter += 1

    debug_print(" Correct and total: ", len(correct_case), len(program_data))

    dead_loop_counter = 0
    to_query = []
    for i, case in tqdm(enumerate(correct_case)):
        debug_print("Start: ", str(case))

        for pred, ans, cot, ques in zip(
            [find_numbers(y) for y in  case.get("new_prediction", [])], 
            case.get("new_answers", []),
            case.get("new_prediction", []),
            case.get("new_questions", [])):
            try:
                if int(pred) == int(float(ans)): # type: ignore
                    case["correct_generations"] = case.get("correct_generations", []) + [ 
                        {
                            "new_question": ques,
                            "new_answer": ans,
                            "new_prediction": cot,
                        }
                     ]
            except Exception as ex:
                print(ex)

        gen_samples = 5
        gen_samples = 20
        if len(case.get("correct_generations", [])) < gen_samples:
            to_query.append( (i, get_perturb_parameter_value_prompt_messages(case) ))
            case["e2e_generation_complete"] = False
        else:
            case["e2e_generation_complete"] = True
            case.pop("new_questions", None)
            case.pop("new_answers", None)

    debug_print("$$ Filtered for questions for which correct answers were generated ...... \n"
                " Queryig for {} / {} cases".format(len(to_query), len(correct_case)))

    outs = query_model_batch_vllm(
        [x[1] for x in to_query],
        max_length=1000,
        use_temp=0.7
        # use_temp=0.00001
    )


    for (i, _), out in zip(to_query, outs):
        new_values = clear_parameter_output(out)
        if len(new_values) == 6:
            new_values = new_values[1:]
        case = correct_case[i]
        case['new_parameters'] = new_values

    for case in tqdm(correct_case):
        # low hanging fruit here to fix. 
        try:
            case['new_answers'] = []
            for parameter in case['new_parameters']:
                program = case['candidate_programs'][0]
                program += call_function_with_args(parameter)
                f = StringIO()
                if 'while True:' in program:
                    dead_loop_counter += 1
                    continue

                s = execute(program)
                case['new_answers'].append(s)
        except Exception as e:
            print(e)
            print(case['new_parameters'])
            continue

    with open(base_path + common_path + common_save_path + new_params_path, 'w') as outfile:
        json.dump(correct_case, outfile, indent=4)
    print("Dead loop counter: ", dead_loop_counter)
    # exit()

def get_rewrite_question_prompt_messages(case):
    prompt_template = PROMPT_DICT['prompt_rewrite_question']
    prompt = prompt_template.format_map(
        {"question": case['question'], "parameters": case['parameters'], "new_parameters": "\n".join(case['new_parameters'])}
    )

    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    return messages

def update_question_with_new_parameters_parallel():
    debug_print("Inside update_question_with_new_parameters function .... ")
    infile = open(base_path + common_path + common_save_path + new_params_path, 'r')
    program_data = json.load(infile)
    print(len(program_data))
    rewrite_questions_messages_list = []
    for i, case in enumerate(program_data):
        if case["e2e_generation_complete"] == False:
            rewrite_questions_messages_list.append(
                (i, get_rewrite_question_prompt_messages(case)))

    outs = query_model_batch_vllm(
        [x[1] for x in rewrite_questions_messages_list],
        use_temp=0.00001,
        max_length=2048)

    # for case, out in tqdm(zip(program_data, outs)):
    for msg, out in zip(rewrite_questions_messages_list, outs):
        new_values = [x.strip() for x in out.split('\n') if len(x) > 0]
        if len(new_values) == 6:
            new_values = new_values[1:]
        case = program_data[msg[0]]
        case['new_questions'] = new_values
        debug_print("----- ", case["question"])
        debug_print("----- ", case["new_questions"])
        debug_print("\n\n" + "-" * 20 + "\n\n")
    outfile = open(base_path + common_path + common_save_path + new_ques_path, 'w')
    json.dump(program_data, outfile, indent=4)

def get_answer_question_prompt_messages(question, model_name='gpt', cot=False):
    if cot:
        prompt_template = PROMPT_DICT['prompt_answer_question_few_shot_cot']
    else:
        prompt_template = PROMPT_DICT['prompt_answer_question']
    prompt = prompt_template.format_map(
        {"question": question}
    )
        
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages

def answer_question_parallel(model_name='gpt', cot=False, repeats=5):
    debug_print("Inside answer_question function .... ")
    infile = open(base_path + common_path + common_save_path + new_ques_path, 'r')
    program_data = json.load(infile)
    print(len(program_data))

    # suranjan:Why is this needed?
    if "prediction" not in program_data[0]:
        answer_question_messages_list = []
        for case in tqdm(program_data):
            messages = get_answer_question_prompt_messages(
                case['question'], model_name=model_name, cot=cot)
            answer_question_messages_list.append(messages)

        outs = query_model_batch_vllm(
            answer_question_messages_list,
            max_length=1000,
            use_temp=0.7
        )

        for case, response in tqdm(zip(program_data, outs)):
            case['prediction'] = response

    for _ in range(repeats):
        id_list = []
        question_list = []
        answer_new_question_messages_list = []
        for i, case in enumerate(program_data):
            x = 0
            if case["e2e_generation_complete"]:
                continue
            for question, prediction, answer in zip(
                case['new_questions'], 
                case.get('new_prediction', ["-50000000" for _ in range(len(case['new_questions']))]) , 
                case['new_answers']):

                predicted_answer = get_last_number(prediction)
                failed = True
                try:
                    failed = int(predicted_answer) != int(float(answer))
                except Exception as ex:
                    print(ex)
                
                if failed:
                    question_list.append(question)
                    id_list.append((i, x))
                    debug_print("New question : ", question, predicted_answer, answer)
                    answer_new_question_messages_list.append(
                        get_answer_question_prompt_messages(
                            question, model_name=model_name, cot=cot)
                    )

                x += 1

        debug_print("Finding answers for {} new questions".format(len(answer_new_question_messages_list)))
        outs = query_model_batch_vllm(
            answer_new_question_messages_list,
            max_length=1000,
            use_temp=0.7
        )

        for id_pair, out in zip(id_list, outs):
            id_1, id_2 = id_pair
            if 'new_prediction' not in program_data[id_1]:
                program_data[id_1]['new_prediction'] = ["" for _ in range(len(program_data[id_1]['new_questions']))]
            program_data[id_1]['new_prediction'][id_2] = out

        with open(base_path + common_path + common_save_path + new_ques_ans_path, 'w') as outfile:
            json.dump(program_data, outfile, indent=4)

def parse_answer(answer):
    if type(answer) is not list:
        if 'answer is' in answer:
            answer = answer.split('answer is')[-1].strip()
        else:
            answer = answer.split(' ')[-1]
        if len(answer) > 0 and answer[-1] == '.':
            answer = answer[0:-1]
        answer = re.sub("[^\d\.]", "", answer)
        return answer
    else:
        answer_freq = {}
        for x in answer:
            parsed_answer = parse_answer(x)
            if parsed_answer not in answer_freq:
                answer_freq[parsed_answer] = 0
            answer_freq[parsed_answer] += 1
        answer_freq = sorted(answer_freq.items(), key=lambda item: item[1], reverse=True)
        return answer_freq[0][0]

def main():
    for _ in range(5):
        generate_new_parameter_value_parallel()

        update_question_with_new_parameters_parallel()

        answer_question_parallel(model_name='llama', cot=True)


if __name__ == "__main__":
    main()

"""

See run_program_parallel.py for this part. 

python3.9 -W ignore scripts/generate_gsm8k_new_samples.py 13 | tee /tmp/output.txt; 

"""


