# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List
import random

def get_seeds(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            datas.append(json.loads(line))
    return datas


def get_augmented_dials(file_path):
    augmented_dials = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            augmented_dials.append(line.strip())
    return augmented_dials


def get_random_indices():
    random_indices = []
    while len(random_indices) < 3:
        index = random.randint(0, 59)
        if index not in random_indices:
            random_indices.append(index)
    return random_indices


def format_instruction(datas, random_indices, augmented_dials, i):
    return f"""### Instruction: Generate dialogue state given dialogues that 'user' is asking 'bot' for recommendation food or travel. I will give you some samples. The 'prev_state' (i.e., previous state) is the dialogue state that determined before the user's last utterance. The 'cur_state' (i.e., current state) is the dialogue state that determined after the user's last utterance. You should generate accurately both 'prev_state' and 'cur_state', following the structure of given samples. \n ### Input: [Dialogue 1] {datas[random_indices[0]]['dialogue']} 'prev_state': {datas[random_indices[0]]['prev_state']} 'cur_state': {datas[random_indices[0]]['cur_state']} \n [Dialogue 2] {datas[random_indices[1]]['dialogue']} 'prev_state': {datas[random_indices[1]]['prev_state']} 'cur_state': {datas[random_indices[1]]['cur_state']} \n [Dialogue 3] {datas[random_indices[2]]['dialogue']} 'prev_state': {datas[random_indices[2]]['prev_state']} 'cur_state': {datas[random_indices[2]]['cur_state']} \n ### Output: [Dialogue 4] {augmented_dials[i]} """


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    seeds = get_seeds('./datas/samples_translation.json')
    augmented_dialogues = get_augmented_dials('./datas/dialogue_augment.txt')
    random_indices = get_random_indices()

    prompts = []
    for i in range(0, 30):
        prompts.append(format_instruction(seeds, random_indices, augmented_dialogues, i))

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
