"""
性能监控模块
提供 API 请求耗时、错误率、系统资源监控
"""
import time
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._request_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "errors": 0,
            "last_request": None,
            "avg_time": 0.0,
            "max_time": 0.0,
            "min_time": float('inf'),
        })
        self._recent_requests: List[Dict] = []
        self._start_time = datetime.now()

    def record_request(self, path: str, method: str, status_code: int, duration: float):
        """记录一次 API 请求"""
        key = f"{method} {path}"

        with self._lock:
            stats = self._request_stats[key]
            stats["count"] += 1
            stats["total_time"] += duration
            stats["errors"] += 1 if status_code >= 400 else 0
            stats["last_request"] = datetime.now().isoformat()
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["max_time"] = max(stats["max_time"], duration)
            stats["min_time"] = min(stats["min_time"], duration)

            # 记录最近的请求
            self._recent_requests.append({
                "path": path,
                "method": method,
                "status_code": status_code,
                "duration": round(duration * 1000, 2),  # ms
                "timestamp": datetime.now().isoformat(),
            })

            # 限制条目数
            if len(self._recent_requests) > self.max_entries:
                self._recent_requests = self._recent_requests[-self.max_entries:]

    def get_stats(self) -> Dict:
        """获取性能统计"""
        with self._lock:
            total_requests = sum(s["count"] for s in self._request_stats.values())
            total_errors = sum(s["errors"] for s in self._request_stats.values())
            uptime = (datetime.now() - self._start_time).total_seconds()

            # 慢请求（>1秒）
            slow_requests = [
                r for r in self._recent_requests
                if r["duration"] > 1000
            ]

            # 按端点排序
            top_endpoints = sorted(
                [{"endpoint": k, **v} for k, v in self._request_stats.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:20]

            return {
                "uptime_seconds": round(uptime),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": f"{total_errors / total_requests * 100:.2f}%" if total_requests > 0 else "0%",
                "slow_request_count": len(slow_requests),
                "recent_slow_requests": slow_requests[-10:],
                "top_endpoints": top_endpoints,
                "system_info": self._get_system_info(),
            }

    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        import os
        import platform

        info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "process_id": os.getpid(),
        }

        try:
            # 内存使用
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS'):
                        mem_kb = int(line.split()[1])
                        info["memory_mb"] = round(mem_kb / 1024, 1)
                        break
        except (FileNotFoundError, IndexError):
            pass

        return info

    def get_recent_errors(self, limit: int = 50) -> List[Dict]:
        """获取最近的错误请求"""
        with self._lock:
            errors = [
                r for r in self._recent_requests
                if r["status_code"] >= 400
            ]
            return errors[-limit:]

    def reset(self):
        """重置统计"""
        with self._lock:
            self._request_stats.clear()
            self._recent_requests.clear()
            self._start_time = datetime.now()


# 全局监控实例
monitor = PerformanceMonitor()


# FastAPI 中间件用装饰器
class Timer:
    """请求计时器"""

    def __init__(self, path: str, method: str):
        self.path = path
        self.method = method
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        status_code = 500 if exc_type else 200
        monitor.record_request(self.path, self.method, status_code, duration)
