import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.multi_img_conversation import Chat, CONV_VISION
from chain_of_thought_imgs import img_descriptions

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def pick_random_file(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter out directories, if any
    files = [file for file in files if os.path.isfile(os.path.join(directory, file))]
    
    if not files:
        print("No files found in the directory.")
        return None
    
    # Choose a random file from the list
    random_file = random.choice(files)
    
    # Return the full path of the randomly chosen file
    return os.path.join(directory, random_file)


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset():
    # reset chatbot, image, text_input, upload_button, chat_state, img_list, img_emb_list, gallery
    return None, \
        gr.update(value=None, interactive=True), \
        gr.update(placeholder='Please upload your image first', interactive=False), \
        gr.update(value="Upload & Start Chat", interactive=True), \
        CONV_VISION.copy(), \
        [], \
        [], \
        []


def upload_img(gr_img, chat_state, img_list, img_emb_list):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, img_list, img_emb_list
    img_list.append(gr_img)
    # upload an image to the chat
    chat.upload_img(gr_img, chat_state, img_emb_list)
    # update image, text_input, upload_button, chat_state, gallery, img_emb_list
    return gr.update(value=None, interactive=False), \
        gr.update(interactive=True, placeholder='Type and press Enter'), \
        gr.update(value="Send more images after sending a message", interactive=False), \
        chat_state, \
        img_list, \
        img_emb_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    # update chatbot, chat_state, image, upload_button
    return chatbot, \
        chat_state, \
        gr.update(interactive=True), \
        gr.update(value="Send more image", interactive=True)

def gradio_FSL(chatbot, conv, gr_img, img_list, img_emb_list, random_img=True):
    if random_img:
        gr_img = pick_random_file("RIM-ONE_DL_images/partitioned_randomly/training_set/glaucoma")
    img_list.append(gr_img)
    examples = img_descriptions.chain_of_thought_imgs
    chat.few_shot_learning_emb(conv, gr_img, examples, img_emb_list)

    prompt = f"""
    Please diganose the image: {examples[0][0]}
    Diagnosis: {examples[0][1]}
    
    Please diganose the image: {examples[1][0]}
    Diagnosis: {examples[1][1]}
    
    Please diganose the image: {examples[2][0]}
    Diagnosis: {examples[2][1]}
    
    Please diagnose the image: {gr_img}
    Diagnosis:
    
    """

    chatbot = chatbot + [[prompt, None]]
    #update chatbot, chat_state (conv), img_list, imb_emb_list
    return chatbot, \
            conv, \
            img_list, \
            img_emb_list,\
            gr.update(value= "Send more images after sending a message", interactive=False)

title = """<h1 align="center">Multi Img Demo of MiniGPT-4</h1>"""
description = """<h3>This is a test of uploading multiple images in one prompt!</h3>"""
article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
"""

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            
            few_shot_learning_button = gr.Button(value="Few Shot Learning on Random Img", interactive=True, variant="primary")
            
            clear = gr.Button("Restart")

            
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State(CONV_VISION.copy())
            img_list = gr.State([])
            img_emb_list = gr.State([])
            gallery = gr.Gallery(label="Uploaded Images", show_label=True) \
                .style(rows=[1], object_fit="scale-down", height="500px", preview=True)
            chatbot = gr.Chatbot(label='MiniGPT-4')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)

   

    upload_button.click(upload_img, [image, chat_state, img_list, img_emb_list],
                        [image, text_input, upload_button, chat_state, gallery, img_emb_list])

    few_shot_learning_button\
        .click(gradio_FSL, [chatbot, chat_state, image, img_list, img_emb_list],
                           [chatbot, chat_state, img_list, img_emb_list, upload_button]) \
        .then(gradio_answer, [chatbot, chat_state, img_emb_list, num_beams, temperature],
                             [chatbot, chat_state, image, upload_button])

    text_input \
        .submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]) \
        .then(gradio_answer,
              [chatbot, chat_state, img_emb_list, num_beams, temperature],
              [chatbot, chat_state, image, upload_button])

    clear.click(gradio_reset,
                None,
                [chatbot, image, text_input, upload_button, chat_state, img_list, img_emb_list, gallery],
                queue=False)

demo.launch(share=True, enable_queue=True)
