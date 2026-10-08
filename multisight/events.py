"""轻量进程内事件总线：CLI 美化输出与 WebUI SSE 实时进度共用同一事件源。

设计沿用 Butian3D 的经验：状态机推进与事件广播解耦，订阅方（控制台 / SSE）
只消费事件，不反向影响管线。history 支持新订阅者回放（页面刷新后不丢进度）。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Iterable


class Bus:
    """线程安全的发布/订阅总线，带历史回放。"""

    def __init__(self, history_limit: int = 500) -> None:
        self._listeners: list[queue.Queue] = []
        self._history: deque[dict] = deque(maxlen=history_limit)
        self._lock = threading.Lock()

    def publish(self, event_type: str, **fields: Any) -> dict:
        """广播一个事件。所有事件自动附 ts；end 事件标志管线终止。"""
        event = {"type": event_type, "ts": time.time(), **fields}
        with self._lock:
            self._history.append(event)
            listeners = list(self._listeners)
        for q in listeners:
            q.put(event)
        return event

    def subscribe(self, replay: bool = True) -> queue.Queue:
        """获取一个订阅队列；replay=True 时先把历史事件灌入（SSE 断线重连友好）。"""
        q: queue.Queue = queue.Queue()
        with self._lock:
            if replay:
                for event in self._history:
                    q.put(event)
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def history(self) -> Iterable[dict]:
        with self._lock:
            return list(self._history)
