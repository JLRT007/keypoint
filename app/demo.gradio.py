import gradio as gr
from model import UniModel
from func import ywqz_count
from func import YWQZ
from ai.bot import ai_advice
model = UniModel()
def process_frame(image):
    canvas, keypoint_res = model.infer_img(image, vis=True)
    return canvas, keypoint_res['boxes_num']

def process_video(video):
    out_path,keypoint_data = model.infer_vid(video)
    evaluator = YWQZ()
    ywqz_num = evaluator.count_complete_reps(keypoint_data)#计算仰卧起坐个数
    advice = evaluator.generate_advice(keypoint_data)
    Botadvice = ai_advice(advice)
    #ywqz_num  = ywqz_count(ywqz_data)
    return out_path,ywqz_num,Botadvice

with gr.Blocks() as demo:
    html_title = '''
                <h1 style="text-align: center; color: #333;">飞桨智姿</h1>
                '''
    gr.HTML(html_title)
    with gr.Tab("图像处理"):
        # Blocks默认设置所有子组件按垂直排列Column
        with gr.Row():
            image_input = gr.Image(label='输入图像')
            with gr.Column():
                image_output = gr.Image(label='检测结果')
                text_image = gr.Textbox(label='行人数量')
        image_button = gr.Button(value="提交")
    with gr.Tab("仰卧起坐"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label='输入图像')
            with gr.Column():
                video_output = gr.Video(label='检测结果')
                text_video = gr.Textbox(label='仰卧起坐个数')
                advice_video = gr.Textbox(label='建议')
        video_button = gr.Button(value="提交")
    image_button.click(process_frame, inputs=image_input, outputs=[image_output, text_image])
    video_button.click(process_video, inputs=video_input, outputs=[video_output,text_video,advice_video])

demo.launch()
