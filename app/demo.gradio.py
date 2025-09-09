import gradio as gr
from model import UniModel
from func import ywqz_count
from func import YWQZ,PlankEvaluator
from ai.bot import ai_advice
model = UniModel()
def process_frame(image):
    canvas, keypoint_res = model.infer_img(image, vis=True)
    return canvas, keypoint_res['boxes_num']

def ywqz_video(video):  #仰卧起坐
    out_path,keypoint_data = model.infer_vid(video)
    evaluator = YWQZ()
    ywqz_num = evaluator.count_complete_reps(keypoint_data)#计算仰卧起坐个数
    advice = evaluator.generate_advice(keypoint_data)
    Botadvice = ai_advice(advice,'ywqz')
    #ywqz_num  = ywqz_count(ywqz_data)
    return out_path,ywqz_num,Botadvice
def pbzc_video(video):  #平板支撑
    valid_frame_count = 0
    valid_frame = 0
    out_path, keypoint_data,fps = model.infer_vid(video)
    #print(frame_timestamp_list)
    evaluator = PlankEvaluator()
    for idx,frame_keypoints in enumerate(keypoint_data):
        eval_result = evaluator.evaluate_frame(frame_keypoints)
        #evaluator.update_timer(eval_result["valid"])
        if eval_result['valid']:
            valid_frame += 1
    standard_duration = 1/fps * valid_frame
    #standard_duration = evaluator.get_current_duration()#获取标准动作的总时长
    advice_list = evaluator.generate_advice()#获取建议
    Botadvice = ai_advice(advice_list, 'pbzc')
    return out_path,standard_duration,Botadvice
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
                video_input1 = gr.Video(label='输入图像')
            with gr.Column():
                video_output1 = gr.Video(label='检测结果')
                text_video1 = gr.Textbox(label='仰卧起坐个数')
                advice_video1 = gr.Textbox(label='建议')
        video_button1 = gr.Button(value="提交")
    with gr.Tab("平板支撑"):
        with gr.Row():
            with gr.Column():
                video_input2 = gr.Video(label='输入图像')
            with gr.Column():
                video_output2 = gr.Video(label='检测结果')
                text_video2 = gr.Textbox(label='平板支撑时间')
                advice_video2 = gr.Textbox(label='建议')
        video_button2 = gr.Button(value="提交")
    image_button.click(process_frame, inputs=image_input, outputs=[image_output, text_image])
    video_button1.click(ywqz_video, inputs=video_input1, outputs=[video_output1,text_video1,advice_video1])
    video_button2.click(pbzc_video, inputs=video_input2, outputs=[video_output2, text_video2, advice_video2])

demo.launch()
