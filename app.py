import os
import time
import concurrent.futures
from google import genai
from google.genai import types
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# 加载环境变量
load_dotenv()

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 配置 Gemini API Client
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash-exp')

if not GEMINI_API_KEY:
    print("警告: 未找到 GEMINI_API_KEY 环境变量，请检查 .env 文件")

client = genai.Client(api_key=GEMINI_API_KEY)

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_gemini_response(index, system_prompt, user_prompt, image_path):
    """
    调用 Google Gemini API (v2 SDK) 生成回复
    """
    try:
        # 读取图片数据
        with open(image_path, 'rb') as f:
            image_data = f.read()
            
        mime_type = 'image/jpeg'
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'

        # 构建 Prompt
        # SDK v2 使用 contents 列表
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=system_prompt), # System prompt usually part of user message or separate system instruction
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    types.Part.from_text(text=user_prompt)
                ]
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
                        with open(gen_filepath, 'wb') as f:
                            f.write(part.inline_data.data)
                            
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
    if 'image' not in request.files:
        return jsonify({"error": "没有上传图片"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
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
    results = []
    # 使用 ThreadPoolExecutor 进行并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(batch_count):
            futures.append(
                executor.submit(generate_gemini_response, i, system_prompt, user_prompt, filepath)
            )
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"status": "error", "error": str(e)})

    # 结果按索引排序
    results.sort(key=lambda x: x.get('index', 0))

    return jsonify({
        "message": "批处理完成",
        "total": batch_count,
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)

