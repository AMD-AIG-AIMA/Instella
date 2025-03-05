import time
import json
import ast
from sys import dont_write_bytecode

# import codegen
import os
import re
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from vllm import LLM, SamplingParams

cntr = 0
base_path = "math_data"
executed_programs_path = "executed_programs.json"
processed_abstractions_path = "processed_abstraction.json"
generated_programs_path = "generated_programs.json"
model_name, model, tokenizer = "Qwen/Qwen2.5-72B-Instruct", None, None
# model_name, model, tokenizer = "meta-llama/Llama-3.2-1B-Instruct", None, None

def initiate(model_name=model_name):
    global model, tokenizer
    model = LLM(model_name, 
                enable_prefix_caching=True,
                tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

def make_vllm_calls(
        messages_list, 
        tokenizer_here=None, 
        max_tokens=256,
        temp=0.7, 
        model_here=None, 
        model_name=model_name):

    debug_print("Running a batch of vllm queries with .... {} ".format(len(messages_list)))

    if model_here is None:
        global model, tokenizer
        if model is None:
            initiate(model_name)
        model_here, tokenizer_here = model, tokenizer

    sampling_params = SamplingParams(
        temperature=temp, 
        max_tokens=max_tokens, 
        stop="<|eot_id|>")
    
    params_list = []
    for msg in messages_list:
        params = {
                "model": model_name,
                "max_tokens": max_tokens,
                "stop": "<|eot_id|>"
            }
        params["messages"] = msg
        params_list.append(params)

    prompts = [
        tokenizer_here.apply_chat_template(params["messages"],
                                        tokenize=False,
                                        add_generation_prompt=True)
                                        for params in params_list]
    
    def split_into_chunks(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    
    prompt_batches = split_into_chunks(prompts, 1000)

    outputs = []
    for curr_prompts_batch in prompt_batches:
        outputs_batch =  model_here.generate(
            curr_prompts_batch,
            sampling_params)
        outputs.extend(outputs_batch)

    all_out = []
    for param, output in zip(params_list, outputs):
        prompt = ""
        output_string = ""
        try:
            prompt = output.prompt
            output_string = output.outputs[0].text
        except Exception as ex:
            print(f"{param} generated an exception: {ex}")
        all_out.append((
            output_string,
            param))

        debug_print("--------------------  \n*****Prompt : {} \n*****Output : {} \n".format(prompt, output_string) )

    return [x[0] for x in all_out]

llama_pipeline = None
DEBUG_MODE = True

def debug_print(*args, **kwargs):
    """Custom print function that prints only when DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

def format_abstraction(question):
    messages = [
        {"role": "system",
         "content": "Identify numerical values in the given question, then replace some of them with Python parameters that are either int or float, so that the resulting abstract question is still answerable with the same general solution as the original question. Follow the the provided examples."},
        {"role": "user",
         "content": "Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?"},
        {"role": "assistant",
         "content": "Becuase this is a math question, we identify all numerical values. We identify: 12 inches, 80 pages, one inch, 6\nAs a result, we can replace\n\"12 inches\" to \"Number of Inches X\" (num_inches_x: int)\n\"80 pages\" to \"Number of Pages Y\" (num_pages_y: int)\n\"one inch\" to \"Number of Inches Z\" (num_inches_z: int)\n\"6\" to \"Number W\" (num_w: int)\nSo the question becomes\nJack has a stack of books that is Number of Inches X thick. He knows from experience that Number of Pages Y is Number of Inches Z thick. If he has Number W books, how many pages is each one on average?\nWith parameters\nnum_inches_x=12, num_pages_y=80, num_inches_z=1, num_w=6"},
        {"role": "user",
         "content": "Benny bought  2 soft drinks for$ 4 each and 5 candy bars. He spent a total of 28 dollars. How much did each candy bar cost?"},
        {"role": "assistant",
         "content": "Becuase this is a math question, we identify all numerical values. We identify: 2, $4, 5, 28 dollars\nAs a result, we can replace\n\"2\" to \"Number X\" (num_x: int)\n\"$4\" to \"Dollar Amount Y\" (dollar_y: int)\n\"5\" to \"Number Z\" (num_z: int)\n\"28 dollars\" to \"Dollar Amount W\" (dollar_w: int)\nSo the question becomes\nBenny bought Number X soft drinks for Dollar Amount Y each and Number Z candy bars. He spent a total of Dollar Amount W dollars. How much did each candy bar cost?\nWith parameters\nnum_x=2, dollar_y=4, num_z=5, dollar_w=28"},
        {"role": "user",
         "content": "Wickham is throwing a huge Christmas party. He invites 30 people. Everyone attends the party, and half of the guests bring a plus one (one other person). He plans to serve a 3-course meal for the guests. If he uses a new plate for every course, how many plates does he need in total for his guests?"},
        {"role": "assistant",
         "content": "Becuase this is a math question, we identify all numerical values. We identify: 30, half, plus one (one other person), 3-course\nAs a result, we can replace\n\"30\" to \"Number X\" (num_x: int)\n\"half\" to \"Fraction Y\" (fraction_y: float)\n\"plus one (one other person)\" to \"Number of Additional People Z\" (num_additional_z: int)\n\"3-course\" to \"Number of Course W\" (num_courses_w: int)\nSo the question becomes\nWickham is throwing a huge Christmas party. He invites Number X people. Everyone attends the party, and Fraction Y of the guests bring Number of Additional People Z. He plans to serve a Number of Courses W meal for the guests. If he uses a new plate for every course, how many plates does he need in total for his guests?\nWith parameters\nnum_x=30, fraction_y=0.5, num_additional_z=1, num_courses_w=3"},
        {"role": "user",
         "content": "John volunteers at a shelter twice a month for 3 hours at a time.  How many hours does he volunteer per year?"},
        {"role": "assistant",
         "content": "Becuase this is a math question, we identify all numerical values. We identify: twice, 3\nAs a result, we can replace\n\"twice\" to \"Number of Occurrences X\" (num_occurrence_x: int)\n\"3\" to \"Number Y\" (num_y: int)\nJohn volunteers at a shelter Number of Occurrences X per month for Number Y hours at a time.  How many hours does he volunteer per year?\nWith parameters\nnum_occurrence_x=2, num_y=3"},
    ]
    messages.append({"role": "user", "content": question})
    return messages

def query_model_batch_vllm(
        formatted_message_list, 
        do_sample=None, 
        max_length=256, 
        use_temp=None, 
        gen_program=None):
    if gen_program is not None and gen_program == True:
        assert max_length > 512

    if do_sample is not None:
        raise NotImplementedError("error .... ")

    if use_temp is None:
        use_temp = 0.7

    return make_vllm_calls(
        formatted_message_list, 
        max_tokens=max_length,
        temp=use_temp)

def process_abstraction_all():
    global cntr
    tgt_path1 = base_path + executed_programs_path
    tgt_path2 = base_path + processed_abstractions_path
    all_objs_list = []
    if os.path.exists(tgt_path1):
        print("tgt_path exists, loading from it .....")
        with open(tgt_path1, 'r') as infile:
            all_objs_list = json.load(infile)
    elif os.path.exists(tgt_path2):
        print("tgt_path exists, loading from it .....")
        with open(tgt_path2, 'r') as infile:
            all_objs_list = json.load(infile)

    # max_objects_list = 30
    max_objects_list = 7470
    if len(all_objs_list) < max_objects_list:
        import datasets
        all_objs_list = datasets.load_dataset( "openai/gsm8k", "main", split="train").to_list()
        all_objs_list = all_objs_list[:max_objects_list]

    debug_print("Processing all the abstraction objects ..... ")
    
    def format_abstraction_for_all(all_objs_list):
        return [format_abstraction(obj["question"]) for obj in all_objs_list]

    def process_response(ret, question):
        response = ret.strip().split("\n")
        replace_map = {}
        masked_question = ""
        parameters = ""
        for i, line in enumerate(response):
            if line.startswith("\""):
                group = re.findall('"([^"]*)"', line)
                if len(group) != 2:
                    continue
                codename = re.findall('\([^"]*\)', line)
                if len(codename) == 0:
                    continue
                codename = codename[-1][1:-1]
                replace_map[group[0].strip()] = (group[1].strip(), codename)
            if line.startswith("So the question becomes") and i < len(response) - 1:
                masked_question = response[i + 1]
            if line.startswith("With parameters") and i < len(response) - 1:
                parameters = response[i + 1]
        para_map = {}
        for k in replace_map:
            para_map[k] = replace_map[k][0] + " ({})".format(replace_map[k][1])
        masked_question = question
        for k in para_map:
            masked_question = masked_question.replace(k, para_map[k])
        # debug_print( str(question_obj) )
        if masked_question == "":
            return None
        debug_print("**** masked_question, replace_map, parameters : ", masked_question, replace_map, parameters)
        return masked_question, replace_map, parameters

    new_objs_list = []
    for obj in all_objs_list:
        if "correct" in obj and obj["correct"] > 0:
            continue
        keys = ['masked_question', 'replacement', 'parameters', 'candidate_programs', 'candidate_program_results', 'correct']
        obj["prev"] = (
            obj.get("prev", []) +  
            [{k: obj[k] for k in keys if k in obj} ])
        for k in keys:
            if k in obj:
                obj.pop(k)
        new_objs_list.append(obj)

    formatted_message_list = format_abstraction_for_all(new_objs_list)
    returns_list = query_model_batch_vllm(formatted_message_list, max_length=2048, 
                                          use_temp=0.7)

    processed = 0
    for obj, ret in zip(new_objs_list, returns_list):
        response = process_response(ret, obj["question"])
        
        if response is not None:
            obj["masked_question"] = response[0]
            obj["replacement"] = response[1]
            obj["parameters"] = response[2]
            processed += 1

    print("Newly Processed / total / original total / cntr : {} / {} / {} / {}".format(
        processed, len(new_objs_list), len(all_objs_list), cntr))
    pth = base_path + processed_abstractions_path
    generated_prgms_pth = base_path + generated_programs_path
    if os.path.exists(generated_prgms_pth):
        pth = generated_prgms_pth
    with open(pth, 'w') as outfile:
        json.dump(all_objs_list, outfile, indent=4)

def format_masked_question(question, replacement_map):
    system_msg = "Write a Python program to solve the given abstract math question. Your program must contain a function called 'answer' that accepts the input parameters as specified in the question."
    example_questions = [
        "Benny bought Number of Soft Drinks X (num_soft_drinks_x: int) for Cost per Soft Drink Y (cost_per_soft_drink_y: int) each and Number of Candy Bars Z (num_candy_bars_z: int). He spent a total of Total Amount Spent W (total_spent_w: int) dollars. How much did each candy bar cost?",
        "Jack has a stack of books that is Total Thickness X (total_thickness_x: int) inches thick. He knows from experience that Pages per Inch Y (pages_per_inch_y: int) is one inch thick. If he has Number of Books Z (num_books_z: int), how many pages is each one on average?",
        "Wickham is throwing a huge Christmas party. He invites Number of Guests X (num_guests_x: int) people. Everyone attends the party, and Fraction Y (fraction_y: float) of the guests bring a plus one (one other person). He plans to serve a Number of Courses Z (num_courses_z: int) meal for the guests. If he uses a new plate for every course, how many plates does he need in total for his guests?",
        "A church has Total Members X (total_members_x: int). Percentage Y (percentage_y: float) are adults. The rest are children. How many children more are there than adults?",
    ]
    x1 = ["'''\nTo find the number of children more than adults, we first calculate the number of adults using the percentage given. The rest of the members are children. The difference between the number of children and adults will give us the desired answer.\n'''\n\ndef answer(total_members_x: int, percentage_y: float) -> int:\n    number_of_adults = int((percentage_y) * total_members_x)\n    number_of_children = total_members_x - number_of_adults\n    difference = number_of_children - number_of_adults\n    return difference"]
    x2 = ["'''\nTo find the number of children more than adults, we first calculate the number of adults using the percentage given. The rest of the members are children. The difference between the number of children and adults will give us the desired answer.\n'''\n\ndef answer(total_members_x: int, percentage_y: float) -> int:\n    number_of_adults = int((percentage_y / 100) * total_members_x)\n    number_of_children = total_members_x - number_of_adults\n    difference = number_of_children - number_of_adults\n    return difference"]
    example_responses = [
        "'''\nTo solve this question, we need to calculate the total cost of the soft drinks and subtract it from the total amount spent to find the total cost spent on candy bars. Then, we divide the total cost spent on candy bars by the number of candy bars to find the cost per candy bar.\n'''\n\ndef answer(num_soft_drinks_x: int, cost_per_soft_drink_y: int, num_candy_bars_z: int, total_spent_w: int) -> float:\n\ttotal_cost_soft_drinks = num_soft_drinks_x * cost_per_soft_drink_y\n\ttotal_cost_candy_bars = total_spent_w - total_cost_soft_drinks\n\tcost_candy_bar = total_cost_candy_bars / num_candy_bars_z\n\treturn cost_candy_bar\n#The program ends here.",
        "'''\nTo solve this question, we first need to calculate the total number of pages in the stack of books by multiplying the total thickness by the number of pages per inch. Then, we divide this total number of pages by the number of books to find the average number of pages per book.\n'''\n\ndef answer(total_thickness_x: int, pages_per_inch_y: int, num_books_z: int) -> float:\n\ttotal_pages = total_thickness_x * pages_per_inch_y\n\taverage_pages_per_book = total_pages / num_books_z\n\treturn average_pages_per_book",
        "'''\nTo solve this question, we need to calculate the total number of guests including those who bring a plus one. Then, we multiply this total number of guests by the number of courses to find out how many plates are needed in total.\n'''\n\ndef answer(num_guests_x: int, fraction_y: float, num_courses_z: int) -> int:\n\ttotal_guests = num_guests_x + int(num_guests_x * fraction_y)\n\ttotal_plates_needed = total_guests * num_courses_z\n\treturn total_plates_needed",
        
        
    ]

    import random
    random_bit = random.choice([0, 1])
    if random_bit == 0:
        example_responses += x1 
    else:
        example_responses += x2

    qa = [x for x in zip(example_questions, example_responses)]
    random.shuffle(qa)
    example_questions, example_responses = [x[0] for x in qa], [x[1] for x in qa]

    # Chagned for o1
    messages = [
        {"role": "system", "content": system_msg}
        # {"role": "user", "content": system_msg}
    ]
    limiter = len(example_responses)
    for i in range(0, limiter):
        messages.append({"role": "user", "content": example_questions[i]})
        messages.append({"role": "assistant", "content": example_responses[i]})
    para_map = {}
    for k in replacement_map:
        para_map[k] = replacement_map[k][0] + " ({})".format(replacement_map[k][1])
    
    keys = [(len(k), k) for k in para_map.keys()]
    keys = [k[1] for k in sorted(keys, reverse=True)]
    for k in keys:
        question = question.replace(k, para_map[k])
    messages.append({"role": "user", "content": question})
    return messages

def clean_runnable_program_simple(program):
    lines = program.split("\n")
    outs_lines = []
    start = False
    for line in lines:
        if line.startswith("def") or line.startswith("'''"):
            start = True
        if not start:
            continue
        if "program ends" in line.lower():
            break
        if "```" in line:
            break
        outs_lines.append(line)
    return "\n".join(outs_lines)

def process_program_generation_all():
    src_path = base_path + processed_abstractions_path
    tgt_path = base_path + generated_programs_path

    all_objs_list = []
    if os.path.exists(tgt_path):
        with open(tgt_path, 'r') as infile:
            all_objs_list = json.load(infile)

    with open(src_path, 'r') as infile:
        all_objs_list_new = json.load(infile)
        if len(all_objs_list_new) > len(all_objs_list):
            all_objs_list = all_objs_list_new

    new_objs_list = []
    for obj in all_objs_list:
        if "correct" in obj and obj["correct"] > 0:
            continue
        new_objs_list.append(obj)

    messages_list = [
        format_masked_question(question_obj["question"], question_obj["replacement"]) 
        for question_obj in new_objs_list]
    returns = query_model_batch_vllm(
        messages_list, 
        use_temp=0.7, 
        # use_temp=1.0, 
        gen_program=True, 
        max_length=2048)

    def clean_runnable_program_simple_wrapper(returns, all_objs_list):
        for ret, obj in zip(returns, all_objs_list):
            obj["candidate_programs"] = (
                obj.get("candidate_programs", []) + 
                [clean_runnable_program_simple(ret)]
            ) 

    clean_runnable_program_simple_wrapper(returns, new_objs_list)

    # Call the execute function to generate the results. 

    debug_print("Dumping the output at / cntr ..... {} / {} ".format(
        base_path + generated_programs_path, cntr))
    with open(base_path + generated_programs_path, 'w') as outfile:
        json.dump(all_objs_list, outfile, indent=4)
    # call the function again. 

def execute(program, parameters):
    program_header = "import math"
    function_call = "predicted_answer = answer({})".format(parameters)
    run_program = program_header + "\n" + program + "\n" + function_call + "\nprint(predicted_answer)\n"
    f_open = open(".execution_gen_file_math_x1.py", "w")
    f_open.flush()
    f_open.write(run_program)
    f_open.flush()
    f_open.close()
    os.system("timeout 30 stdbuf -oL python -W ignore .execution_gen_file_math_x1.py > .execution_gen_file_math_output_x1.txt; cat .execution_gen_file_math_output_x1.txt")
    result = open(".execution_gen_file_math_output_x1.txt").read().strip()
    os.system("rm .execution_gen_file_math_x1.py; rm .execution_gen_file_math_output_x1.txt")
    return result

def execute_programs_from_original(question_obj):
    debug_print("Inside execute_programs_from_original function .... ")
    entry_key = "candidate_programs"
    if entry_key not in question_obj:
        return question_obj
    results = []
    for program in question_obj[entry_key]:
        print(program)
        print(question_obj)
        result = execute(program, question_obj["parameters"])
        results.append(result)
    question_obj["candidate_program_results"] = results

    cnt = 0
    for x in results:
        try:
            if float(x) == float(question_obj["answer"].split()[-1]):
                cnt += 1
        except Exception as ex:
            continue

    question_obj["correct"] = cnt * 100 / len(results)

    debug_print(question_obj["answer"], question_obj["correct"])
    debug_print("Results: ", str(results), "\n\n-----------------------------\n\n")
    return question_obj

def execute_programs_from_original_wrapper():
    with open(base_path + generated_programs_path, 'r') as infile:
        all_objs_list = json.load(infile)

    for obj in all_objs_list:
        if "correct" in obj and obj["correct"] > 0:
            continue
        execute_programs_from_original(obj)

    with open(base_path + executed_programs_path, 'w') as outfile:
        json.dump(all_objs_list, outfile, indent=4)

def pipeline_parallel():
    global cntr

    for _ in range(30):
        process_abstraction_all()
        for _ in range(5):
            cntr += 1
            process_program_generation_all()
            execute_programs_from_original_wrapper()
    
if __name__ == "__main__":
    pipeline_parallel()

"""
python3.9 -W ignore scripts/generate_gsm8k_programs.py | tee /tmp/output.txt; 

"""
