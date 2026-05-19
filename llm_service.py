import requests
import logging
from typing import Optional, Dict, List, Iterator, Generator
import json
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, CLOUD_API_KEY, CLOUD_API_BASE

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.ollama_base_url = OLLAMA_BASE_URL
        self.default_model = OLLAMA_MODEL
        self.cloud_api_key = CLOUD_API_KEY
        self.cloud_api_base = CLOUD_API_BASE
        logger.info(f"LLM 服务初始化完成 - Ollama: {self.ollama_base_url}, 模型: {self.default_model}")

    def check_ollama_health(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama 服务不可用: {str(e)}")
            return False

    def get_available_models(self) -> List[Dict]:
        """获取可用的 Ollama 模型列表（带详细信息）"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [
                    {
                        "name": m["name"],
                        "size": m.get("size", 0),
                        "modified_at": m.get("modified_at", "")
                    }
                    for m in models
                ]
            return []
        except Exception as e:
            logger.error(f"获取模型列表失败: {str(e)}")
            return []

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """获取指定模型的详细信息"""
        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/show",
                json={"name": model_name},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"获取模型信息失败: {str(e)}")
            return None

    def select_smartest_model(self) -> str:
        """自动选择最大的模型（通常是最聪明的）"""
        models = self.get_available_models()
        if not models:
            logger.warning("未找到可用模型，使用默认模型")
            return self.default_model

        sorted_models = sorted(models, key=lambda x: x.get("size", 0), reverse=True)
        smartest = sorted_models[0]["name"]
        logger.info(f"Auto 模式选择模型: {smartest} (size: {sorted_models[0].get('size', 0)})")
        return smartest

    def generate_local(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context: Optional[List[int]] = None
    ) -> Dict:
        """调用本地 Ollama 模型生成文本"""
        try:
            model_name = model or self.default_model
            
            if context:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "context": context,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            else:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": model_name,
                    "context": result.get("context", []),
                    "source": "local"
                }
            else:
                logger.error(f"Ollama 请求失败: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"请求失败: {response.status_code}",
                    "model": model_name,
                    "source": "local"
                }
                
        except requests.exceptions.Timeout:
            logger.error("Ollama 请求超时")
            return {
                "success": False,
                "error": "请求超时，请检查模型是否正在加载",
                "model": model or self.default_model,
                "source": "local"
            }
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model": model or self.default_model,
                "source": "local"
            }

    def generate_local_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context: Optional[List[int]] = None
    ) -> Generator[str, None, Dict]:
        """流式调用本地 Ollama 模型生成文本"""
        try:
            model_name = model or self.default_model
            
            if context:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "context": context,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            else:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            
            with requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                if response.status_code == 200:
                    full_response = ""
                    context_value = None
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                token = data.get("response", "")
                                if token:
                                    full_response += token
                                    yield token

                                if "context" in data:
                                    context_value = data["context"]
                            except json.JSONDecodeError:
                                continue

                    yield None
                    
                    result = {
                        "success": True,
                        "response": full_response,
                        "model": model_name,
                        "context": context_value or [],
                        "source": "local"
                    }
                    return result
                else:
                    logger.error(f"Ollama 流式请求失败: {response.status_code} - {response.text}")
                    yield f"错误: 请求失败 ({response.status_code})"
                    yield None
                    return {
                        "success": False,
                        "error": f"请求失败: {response.status_code}",
                        "model": model_name,
                        "source": "local"
                    }
                
        except requests.exceptions.Timeout:
            logger.error("Ollama 流式请求超时")
            yield "错误: 请求超时，请检查模型是否正在加载"
            yield None
            return {
                "success": False,
                "error": "请求超时",
                "model": model or self.default_model,
                "source": "local"
            }
        except Exception as e:
            logger.error(f"流式生成失败: {str(e)}")
            yield f"错误: {str(e)}"
            yield None
            return {
                "success": False,
                "error": str(e),
                "model": model or self.default_model,
                "source": "local"
            }

    def generate_cloud(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict:
        """调用云端 API 生成文本（支持 OpenAI 兼容格式）"""
        if not self.cloud_api_key:
            logger.warning("未配置云端 API Key")
            return {
                "success": False,
                "error": "未配置云端 API Key",
                "source": "cloud"
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.cloud_api_key}",
                "Content-Type": "application/json"
            }
            
            # 根据 API 端点选择合适的模型和路径
            if "tbox.cn" in self.cloud_api_base or "ant-ling.com" in self.cloud_api_base:
                # 蚂蚁百灵 API（Ling-2.6-1T / Ling-2.6-flash）
                api_url = f"{self.cloud_api_base.rstrip('/')}/chat/completions"
                # 使用百灵支持的模型
                if model == "deepseek-chat" or model == "auto":
                    model = "Ling-2.6-1T"
            elif self.cloud_api_base.endswith("/v1/") or self.cloud_api_base.endswith("/v1"):
                # URL 已经包含 /v1/
                api_url = f"{self.cloud_api_base.rstrip('/')}/chat/completions"
            else:
                # 标准 OpenAI 兼容格式 - 需要添加 v1 前缀
                api_url = f"{self.cloud_api_base.rstrip('/')}/v1/chat/completions"
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.info(f"调用云端 API: {api_url}, 模型: {model}")
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"云端 API 原始响应: {result}")
                
                # 支持 OpenAI 兼容格式: result["choices"][0]["message"]["content"]
                # 也可能直接返回 result["response"] 或其他格式
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                elif "response" in result:
                    content = result["response"]
                elif "text" in result:
                    content = result["text"]
                else:
                    # 尝试将整个结果转为字符串
                    content = str(result)
                
                return {
                    "success": True,
                    "response": content,
                    "model": model,
                    "source": "cloud",
                    "usage": result.get("usage", {})
                }
            else:
                logger.error(f"云端 API 请求失败: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API 请求失败: {response.status_code}",
                    "source": "cloud"
                }
                
        except Exception as e:
            logger.error(f"云端生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source": "cloud"
            }

    def generate_cloud_stream(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Generator[str, None, Dict]:
        """流式调用云端 API 生成文本（支持 OpenAI 兼容格式）"""
        if not self.cloud_api_key:
            logger.warning("未配置云端 API Key")
            yield "错误: 未配置云端 API Key"
            yield None
            return {
                "success": False,
                "error": "未配置云端 API Key",
                "source": "cloud"
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.cloud_api_key}",
                "Content-Type": "application/json"
            }
            
            if "tbox.cn" in self.cloud_api_base or "ant-ling.com" in self.cloud_api_base:
                api_url = f"{self.cloud_api_base.rstrip('/')}/chat/completions"
                if model == "deepseek-chat" or model == "auto":
                    model = "Ling-2.6-1T"
            elif self.cloud_api_base.endswith("/v1/") or self.cloud_api_base.endswith("/v1"):
                api_url = f"{self.cloud_api_base.rstrip('/')}/chat/completions"
            else:
                api_url = f"{self.cloud_api_base.rstrip('/')}/v1/chat/completions"
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            logger.info(f"调用云端流式 API: {api_url}, 模型: {model}")
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    token = data['choices'][0].get('delta', {}).get('content', '')
                                    if token:
                                        full_response += token
                                        yield token
                        except json.JSONDecodeError:
                            continue
                
                yield None
                
                return {
                    "success": True,
                    "response": full_response,
                    "model": model,
                    "source": "cloud"
                }
            else:
                logger.error(f"云端流式 API 请求失败: {response.status_code} - {response.text}")
                yield f"错误: API 请求失败 ({response.status_code})"
                yield None
                return {
                    "success": False,
                    "error": f"API 请求失败: {response.status_code}",
                    "source": "cloud"
                }
                
        except Exception as e:
            logger.error(f"云端流式生成失败: {str(e)}")
            yield f"错误: {str(e)}"
            yield None
            return {
                "success": False,
                "error": str(e),
                "source": "cloud"
            }

    def chat(
        self,
        prompt: str,
        use_cloud: bool = False,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict:
        """统一的聊天接口"""
        if use_cloud and self.cloud_api_key:
            return self.generate_cloud(prompt, temperature=temperature)
        else:
            return self.generate_local(prompt, model=model, temperature=temperature)


llm_service = LLMService()
