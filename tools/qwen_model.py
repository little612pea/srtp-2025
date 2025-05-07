from transformers import AutoModel, AutoTokenizer
import torch

# model setting
def get_qwen_model():
    model_path = 'OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448'

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
    image_processor = model.get_vision_tower().image_processor

    mm_llm_compress = False # use the global compress or not
    if mm_llm_compress:
        model.config.mm_llm_compress = True
        model.config.llm_compress_type = "uniform0_attention"
        model.config.llm_compress_layer_list = [4, 18]
        model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
    else:
        model.config.mm_llm_compress = False

    # evaluation setting
    max_num_frames = 512
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )
    return model, tokenizer, image_processor, max_num_frames, generation_config

# evaluation setting
def get_qwen_ans(video_path,prompt,model,tokenizer,max_num_frames,generation_config,chat_history=None):
    # multi-turn conversation
    output, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=prompt, chat_history=chat_history, return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)
    return output

if __name__ == "__main__":
    video_path = "./test.mp4"
    model, tokenizer, image_processor, max_num_frames, generation_config = get_qwen_model()
    
    # single-turn conversation
    question1 = "please generate detailed commentary for the match in the video, specifying how the two players gained each point. please describe in detail how the score changed over time."
    output1, chat_history = get_qwen_ans(video_path, question1, model, tokenizer, max_num_frames, generation_config)
    print(output1)
    
    # multi-turn conversation
    question2 = "How many people appear in the video?"
    output2, chat_history = get_qwen_ans(video_path, question2, model, tokenizer, max_num_frames, generation_config, chat_history=chat_history)
    print(output2)