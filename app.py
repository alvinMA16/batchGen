import os
import time
import base64
import concurrent.futures
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

# 加载环境变量
load_dotenv()

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
VIDEO_FOLDER = 'videos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

# 配置 Gemini API Client
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash-exp')

if not GEMINI_API_KEY:
    print("警告: 未找到 GEMINI_API_KEY 环境变量，请检查 .env 文件")

client = genai.Client(api_key=GEMINI_API_KEY)

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_gemini_response(index, system_prompt, user_prompt, image_paths):
    """
    调用 Google Gemini API (v2 SDK) 生成回复
    """
    try:
        parts = [types.Part.from_text(text=system_prompt)]

        # 处理所有图片
        for image_path in image_paths:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            mime_type = 'image/jpeg'
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith('.webp'):
                mime_type = 'image/webp'
            
            parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))

        parts.append(types.Part.from_text(text=user_prompt))

        # 构建 Prompt
        # SDK v2 使用 contents 列表
        contents = [
            types.Content(
                role="user",
                parts=parts
            )
        ]
        
        # 如果模型支持系统指令，也可以放在 config 中，但为了通用性放在 User Message 中
        
        # 调用生成
        # 关键：添加 image_config 以启用图像生成
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    )
                ]
                # 如果明确需要生成图片，某些模型可能需要 image_config
                # 但对于多模态通用模型，通常根据 Prompt 自动决定
                # 为了稳妥，我们不强制加 image_config，除非我们确定是在用专门的 image model
            )
        )

        result_content = ""
        
        # 解析响应
        if response.candidates and response.candidates[0].content.parts:
             for part in response.candidates[0].content.parts:
                # 1. 文本
                if part.text:
                    result_content += part.text + "\n"
                
                # 2. 内嵌图像 (inline_data)
                if part.inline_data:
                    try:
                        print(f"[DEBUG] Received inline data for index {index}. MimeType: {part.inline_data.mime_type}")
                        
                        timestamp = int(time.time() * 1000)
                        ext = 'png'
                        if part.inline_data.mime_type == 'image/jpeg':
                            ext = 'jpg'
                        elif part.inline_data.mime_type == 'image/webp':
                            ext = 'webp'
                            
                        gen_filename = f"gen_{index}_{timestamp}.{ext}"
                        gen_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gen_filename)
                        
                        # 保存图片
                        # SDK v2 中，inline_data.data 通常是 bytes
                        data_to_write = part.inline_data.data
                        print(f"[DEBUG] Data type: {type(data_to_write)}")
                        
                        if data_to_write:
                             print(f"[DEBUG] Data size: {len(data_to_write)} bytes")
                             if isinstance(data_to_write, bytes):
                                 print(f"[DEBUG] First 20 bytes (hex): {data_to_write[:20].hex()}")
                                 
                                 # Check if data is Base64 encoded bytes (common issue with some APIs)
                                 # PNG starts with 'iVBOR', JPEG with '/9j/', WebP with 'UklGR'
                                 # Converting to bytes comparison
                                 if data_to_write.startswith(b'iVBOR') or \
                                    data_to_write.startswith(b'/9j/') or \
                                    data_to_write.startswith(b'UklGR'):
                                     print("[DEBUG] Detected Base64 encoded bytes, attempting to decode...")
                                     try:
                                         decoded_data = base64.b64decode(data_to_write)
                                         print(f"[DEBUG] Base64 decoded successfully. Size: {len(data_to_write)} -> {len(decoded_data)}")
                                         data_to_write = decoded_data
                                     except Exception as b64_err:
                                         print(f"[DEBUG] Base64 decode failed: {b64_err}")
                        else:
                             print(f"[DEBUG] Data is empty!")

                        with open(gen_filepath, 'wb') as f:
                            f.write(data_to_write)
                            f.flush() # Ensure data is written
                            
                        print(f"[DEBUG] Image saved to: {gen_filepath}")
                        result_content += f"\n![Generated Image](/uploads/{gen_filename})\n"
                    except Exception as img_err:
                         print(f"Error saving image: {img_err}")
                         result_content += f"\n[图片保存失败: {str(img_err)}]\n"
        
        if not result_content:
             result_content = "生成成功，但没有返回可见内容 (Text or Image)."

        return {
            "index": index + 1,
            "result": result_content,
            "status": "success"
        }

    except Exception as e:
        print(f"Error in generation {index}: {e}")
        return {
            "index": index + 1,
            "result": f"生成出错: {str(e)}",
            "status": "error"
        }

def generate_video_task(index, prompt, first_frame_path, last_frame_path, duration):
    """
    Call Google Veo API to generate video
    """
    try:
        # Load images as bytes
        with open(first_frame_path, 'rb') as f:
            first_frame_bytes = f.read()
        
        with open(last_frame_path, 'rb') as f:
            last_frame_bytes = f.read()

        # Determine MIME type
        first_mime = 'image/jpeg'
        if first_frame_path.lower().endswith('.png'):
            first_mime = 'image/png'
        elif first_frame_path.lower().endswith('.webp'):
            first_mime = 'image/webp'

        last_mime = 'image/jpeg'
        if last_frame_path.lower().endswith('.png'):
            last_mime = 'image/png'
        elif last_frame_path.lower().endswith('.webp'):
            last_mime = 'image/webp'

        # Create types.Image objects
        first_image = types.Image(
            image_bytes=first_frame_bytes,
            mime_type=first_mime
        )
        last_image = types.Image(
            image_bytes=last_frame_bytes,
            mime_type=last_mime
        )

        # Note: duration is not directly supported in generate_videos currently (as per typical API),
        # but we can include it in the prompt.
        full_prompt = f"{prompt} (Duration: {duration} seconds. Transition to the provided end frame.)"

        print(f"Starting video generation {index} with prompt: {full_prompt}")
        
        # 如果报错 GenerateVideosConfig last_frame Extra inputs are not permitted
        # 说明当前 SDK 版本的 GenerateVideosConfig 不支持 last_frame 字段。
        # 我们暂时只传入 image (起始帧)，并将尾帧意图放入 prompt 中。
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=full_prompt,
            image=first_image,
            # config=types.GenerateVideosConfig(
            #     last_frame=last_image
            # ),
        )
        
        # Poll the operation status
        while not operation.done:
            print(f"Waiting for video generation {index} to complete...")
            time.sleep(5)
            operation = client.operations.get(operation)
            
            # Download the video
        if operation.response and operation.response.generated_videos:
            video_resource = operation.response.generated_videos[0]
            
            timestamp = int(time.time() * 1000)
            video_filename = f"gen_video_{index}_{timestamp}.mp4"
            video_filepath = os.path.join(app.config['VIDEO_FOLDER'], video_filename)
            
            # Ensure client.files.download is called if necessary, then save.
            # Based on snippet:
            client.files.download(file=video_resource.video)
            video_resource.video.save(video_filepath)
            
            print(f"Generated video saved to {video_filepath}")

            return {
                "index": index + 1,
                "result": f"Video Generated based on: {prompt}",
                "video_url": f"/videos/{video_filename}",
                "status": "success"
            }
        else:
            return {
                "index": index + 1,
                "result": "Video generation operation completed but no video resource found.",
                "status": "error"
            }

    except Exception as e:
        print(f"Error in video generation {index}: {e}")
        return {
            "index": index + 1,
            "result": f"Video generation error: {str(e)}",
            "status": "error"
        }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/videos/<filename>')
def video_file(filename):
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

@app.route('/api/history')
def get_history():
    files = []
    try:
        # 获取目录下的所有文件
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # 获取修改时间
                timestamp = os.path.getmtime(filepath)
                files.append({
                    'filename': filename,
                    'url': f'/uploads/{filename}',
                    'timestamp': timestamp
                })
        
        # 按时间倒序排列 (最新的在前)
        files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_generate', methods=['POST'])
def batch_generate():
    # 1. 校验图片
    if 'images' not in request.files:
        return jsonify({"error": "没有上传图片"}), 400
    
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    filepaths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filepaths.append(filepath)
    
    if not filepaths:
        return jsonify({"error": "文件类型不支持"}), 400

    # 2. 获取参数
    data = request.form
    system_prompt = data.get('system_prompt', '')
    user_prompt = data.get('user_prompt', '')
    try:
        batch_count = int(data.get('batch_count', 1))
        # 限制最大批次防止服务器过载
        if batch_count > 20: 
            batch_count = 20
    except ValueError:
        batch_count = 1

    # 3. 并发执行生成任务
    # 使用 stream_with_context 实现流式响应
    def generate_stream():
        # 降低并发数到 3，避免触发速率限制
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(batch_count):
                # 稍微延迟提交，避免瞬间并发过高
                if i > 0:
                    time.sleep(0.5)
                futures.append(
                    executor.submit(generate_gemini_response, i, system_prompt, user_prompt, filepaths)
                )
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    yield json.dumps(result) + "\n"
                except Exception as e:
                    error_res = {"status": "error", "error": str(e)}
                    yield json.dumps(error_res) + "\n"

    return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')

@app.route('/api/video_generate', methods=['POST'])
def video_generate():
    # 1. 校验图片
    if 'first_frame' not in request.files or 'last_frame' not in request.files:
        return jsonify({"error": "缺少首帧或尾帧图片"}), 400
    
    first_file = request.files['first_frame']
    last_file = request.files['last_frame']
    
    if first_file.filename == '' or last_file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if not (allowed_file(first_file.filename) and allowed_file(last_file.filename)):
        return jsonify({"error": "文件类型不支持"}), 400

    # 保存上传的文件到 videos 目录，避免出现在历史图片区
    first_filename = secure_filename(f"v_first_{int(time.time())}_{first_file.filename}")
    last_filename = secure_filename(f"v_last_{int(time.time())}_{last_file.filename}")
    
    first_filepath = os.path.join(app.config['VIDEO_FOLDER'], first_filename)
    last_filepath = os.path.join(app.config['VIDEO_FOLDER'], last_filename)
    
    first_file.save(first_filepath)
    last_file.save(last_filepath)

    # 2. 获取参数
    data = request.form
    prompt = data.get('prompt', '')
    try:
        batch_count = int(data.get('batch_count', 1))
        if batch_count > 5: batch_count = 5
    except ValueError:
        batch_count = 1
        
    try:
        duration = int(data.get('duration', 5))
    except ValueError:
        duration = 5

    # 3. 并发执行生成任务
    # 使用 stream_with_context 流式输出视频结果
    def generate_video_stream():
        # Video generation is slower, limit workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for i in range(batch_count):
                if i > 0:
                    time.sleep(1.0) # 视频生成更耗时，增加间隔
                futures.append(
                    executor.submit(generate_video_task, i, prompt, first_filepath, last_filepath, duration)
                )
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    yield json.dumps(result) + "\n"
                except Exception as e:
                    error_res = {"status": "error", "error": str(e)}
                    yield json.dumps(error_res) + "\n"

    return Response(stream_with_context(generate_video_stream()), mimetype='application/x-ndjson')

if __name__ == '__main__':
    app.run(debug=True, port=5001)

