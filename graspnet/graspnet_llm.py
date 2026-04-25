"""GraspNet LLM 交互服务 - 简化版

提供对话式交互界面，集成了:
- Qwen LLM 语义理解
- ZED 摄像头环境描述
- 原有抓取流程调用
"""

import os
import sys
import base64
import argparse
from typing import Optional, List, Dict

import cv2
import numpy as np
import requests
import gradio as gr

QWEN_MODELS = {
    "qwen-vl-plus": "qwen-vl-plus",
    "qwen-vl-max": "qwen-vl-max",
    "qwen-vl-max-longcontext": "qwen-vl-max-longcontext",
    "qwen-plus": "qwen-plus (不支持视觉)",
    "qwen-turbo": "qwen-turbo (不支持视觉)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="GraspNet LLM 交互服务")
    parser.add_argument("--api_key", type=str, default="sk-517729bb1fa54fcaa8f3077a87438dea",
                        help="阿里云 DashScope API Key")
    parser.add_argument("--model", type=str, default="qwen-vl-plus",
                        help="Qwen 模型选择 (视觉任务请用 qwen-vl 系列)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 服务端口")
    parser.add_argument("--output_dir", type=str, default="runs/llm_grasp",
                        help="输出目录")
    return parser.parse_args()


class QwenLLMClient:
    """Qwen API 客户端"""

    def __init__(self, api_key: str, model: str = "qwen-vl-plus"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List[Dict], temperature: float = 0.7,
             max_tokens: int = 2000) -> str:
        """发送对话请求到 Qwen"""
        if not self.api_key:
            return "错误: 未设置 API Key，请先在设置中输入"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            print(f"[调试] 发送请求到 Qwen, 模型: {self.model}")
            print(f"[调试] 消息数量: {len(messages)}")

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )

            print(f"[调试] 响应状态码: {response.status_code}")

            if response.status_code == 401:
                return "错误: API Key 无效或已过期，请检查后重新设置"
            elif response.status_code == 403:
                return "错误: 没有权限访问该模型，请确认 API Key 有权限"
            elif response.status_code == 400:
                try:
                    error_msg = response.json().get("error", {}).get("message", response.text)
                except:
                    error_msg = response.text
                return f"错误: 请求参数错误 - {error_msg}"

            response.raise_for_status()
            result = response.json()
            print(f"[调试] 响应内容: {str(result)[:500]}")

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            elif "error" in result:
                return f"API 错误: {result['error'].get('message', '未知错误')}"
            else:
                return f"未知响应格式: {result}"

        except requests.exceptions.Timeout:
            return "错误: 请求超时，请稍后重试"
        except requests.exceptions.RequestException as e:
            return f"错误: 网络请求失败 - {str(e)}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"错误: {str(e)}"


class GraspNetLLMService:
    """GraspNet LLM 服务主类"""

    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm_client: Optional[QwenLLMClient] = None
        if args.api_key:
            self.llm_client = QwenLLMClient(args.api_key, args.model)

        self.conversation_history: List[Dict[str, str]] = []
        self.current_image = None

        self.system_prompt = """你是一个智能机械臂抓取系统的助手。你的功能包括：

1. **环境描述**：当用户请求查看环境时，描述摄像头前的场景内容
2. **抓取建议**：根据场景描述，建议可以抓取的物体
3. **执行控制**：响应用户的抓取请求
4. **对话交互**：回答用户关于系统功能的问题

请用中文回答，保持简洁但信息丰富。"""

    def set_api_key(self, api_key: str, model: str):
        """设置 API Key"""
        if api_key:
            self.args.api_key = api_key
            self.args.model = model
            self.llm_client = QwenLLMClient(api_key, model)
            return f"✓ API Key 已设置，使用模型: {model}"
        return "API Key 不能为空"

    def capture_environment(self):
        """捕获环境图像"""
        try:
            import pyzed.sl as sl
            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
            init_params.coordinate_units = sl.UNIT.METER
            init_params.sdk_verbose = False
            init_params.camera_resolution = sl.RESOLUTION.HD720

            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                return None, f"ZED 相机打开失败: {err}"

            image = sl.Mat()
            runtime = sl.RuntimeParameters()

            grabbed = False
            for i in range(15):
                if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                    grabbed = True
                    break

            if not grabbed:
                zed.close()
                return None, "ZED 相机抓取失败"

            zed.retrieve_image(image, sl.VIEW.LEFT)
            color_bgra = image.get_data()
            zed.close()

            if color_bgra is None or color_bgra.size == 0:
                return None, "ZED 返回空图像"

            if color_bgra.ndim == 3 and color_bgra.shape[2] == 4:
                color_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
            else:
                color_bgr = color_bgra.copy()

            if color_bgr.size == 0:
                return None, "转换后图像为空"

            self.current_image = color_bgr
            img_path = os.path.join(self.output_dir, "current_scene.jpg")
            cv2.imwrite(img_path, color_bgr)
            print(f"[调试] 图像已保存到: {img_path}, 尺寸: {color_bgr.shape}")
            return color_bgr, None

        except ImportError:
            return None, "pyzed 模块不可用"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"捕获失败: {str(e)}"

    def describe_environment(self) -> str:
        """描述当前环境"""
        img, err = self.capture_environment()
        if err:
            return f"无法捕获环境: {err}"
        if img is None:
            return "无法获取环境图像"

        if self.current_image is None or self.current_image.size == 0:
            return "当前没有有效图像"

        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, buffer = cv2.imencode('.jpg', self.current_image, encode_param)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            print(f"[调试] 发送图像到 LLM, base64 长度: {len(image_base64)}, 模型: {self.args.model}")

            prompt = """请详细描述这张图片中的场景内容：
1. 场景中有哪些物体？
2. 物体的空间位置关系是怎样的？
3. 物体有什么显著特征？
4. 是否有适合抓取的物体？

请用中文回答，简洁但信息丰富。"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ]

            if self.llm_client:
                response = self.llm_client.chat(messages)
                print(f"[调试] LLM 响应: {response[:200] if len(response) > 200 else response}")
                return response
            else:
                return "错误: LLM 未初始化，请先在设置中输入 API Key"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"描述失败: {str(e)}"

    def process_command(self, user_input: str) -> str:
        """处理用户命令"""
        if not self.llm_client:
            return "错误: LLM 未初始化，请先在设置中输入 API Key"

        user_input_lower = user_input.lower()

        if any(k in user_input_lower for k in ["查看环境", "查看场景", "描述环境", "当前有什么", "看看"]):
            description = self.describe_environment()
            return f"【环境捕获成功】\n\n{description}"

        if "抓取" in user_input or "执行" in user_input:
            try:
                import subprocess
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "zed_piper_grasp.py")
                result = subprocess.run(
                    [sys.executable, script_path,
                     "--execute",
                     "--output_dir", self.output_dir],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=300
                )
                if result.returncode == 0:
                    return f"【抓取执行成功】\n{result.stdout[-500:]}"
                else:
                    return f"【抓取执行失败】\n{result.stderr[-300:]}"
            except subprocess.TimeoutExpired:
                return "错误: 抓取执行超时"
            except Exception as e:
                return f"错误: {str(e)}"

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})

        response = self.llm_client.chat(messages)
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def chat_fn(self, user_input: str, history: list):
        """Gradio 对话回调"""
        if not user_input.strip():
            return "", history

        response = self.process_command(user_input)
        history.append((user_input, response))
        return "", history

    def build_ui(self):
        """构建简化版 Gradio UI"""
        with gr.Blocks(title="GraspNet LLM") as demo:
            gr.Markdown("# 🤖 GraspNet LLM 控制中心")
            gr.Markdown("**注意**: 视觉任务（如环境描述）请使用 `qwen-vl-plus` 或 `qwen-vl-max` 模型")

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500, label="对话")
                    msg_input = gr.Textbox(
                        label="输入指令",
                        placeholder="输入指令，例如：查看环境、描述场景、执行抓取...",
                        lines=2
                    )
                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清除")

                with gr.Column(scale=1):
                    gr.Markdown("## 设置")
                    api_key_input = gr.Textbox(
                        label="DashScope API Key",
                        value=self.args.api_key,
                        type="password",
                        placeholder="输入您的 API Key"
                    )
                    model_dropdown = gr.Dropdown(
                        label="模型 (视觉任务用 qwen-vl 系列)",
                        choices=list(QWEN_MODELS.keys()),
                        value=self.args.model
                    )
                    update_btn = gr.Button("更新设置", variant="primary")
                    status_output = gr.Textbox(label="状态", lines=2)

                    gr.Markdown("---")
                    gr.Markdown("### 快捷指令")
                    gr.Examples(
                        examples=[
                            ["查看环境"],
                            ["描述当前场景"],
                            ["执行抓取"],
                        ],
                        inputs=msg_input
                    )

            def update_setting(api_key, model):
                return self.set_api_key(api_key, model)

            update_btn.click(
                fn=update_setting,
                inputs=[api_key_input, model_dropdown],
                outputs=[status_output]
            )

            submit_btn.click(
                fn=self.chat_fn,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )

            msg_input.submit(
                fn=self.chat_fn,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )

            clear_btn.click(
                fn=lambda: ([], ""),
                inputs=[],
                outputs=[chatbot, msg_input]
            )

        return demo

    def run(self):
        """启动服务"""
        demo = self.build_ui()
        demo.launch(
            server_name="0.0.0.0",
            server_port=self.args.port
        )


def main():
    args = parse_args()

    print("=" * 50)
    print("GraspNet LLM 控制中心 (简化版)")
    print("=" * 50)
    print(f"API Key: {'已设置' if args.api_key else '未设置'}")
    print(f"模型: {args.model}")
    print(f"端口: {args.port}")
    print("=" * 50)
    print("提示: 视觉任务请使用 qwen-vl-plus 或 qwen-vl-max 模型")
    print("=" * 50)

    service = GraspNetLLMService(args)
    service.run()


if __name__ == "__main__":
    main()
