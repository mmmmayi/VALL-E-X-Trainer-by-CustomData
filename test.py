from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
checkpoint='/scratch/users/astar/ares/ma_yi/output/vallex/exp/stage-2/epoch-1.pt'
preload_models(checkpoint)

# generate audio from text
text = "读书使人进步,学习让我们的眼界更加开阔"
text_prompt = "笔记它只是一个工具就是最终的目的是吸收这些知识"
audio_prompt = "/home/users/astar/ares/ma_yi/code/SLAM-LLM/examples/vallex/demo/zh_prompt.wav" 
target_lang = "zh"
prompt_lang = "zh"
model_home = "/scratch/users/astar/ares/ma_yi/output/vallex/"
audio_array = generate_audio(text, model_home, text_prompt, audio_prompt, target_lang=target_lang, prompt_lang=prompt_lang)

# save audio to disk
#write_wav("zh2zh.wav", 24000, audio_array)
