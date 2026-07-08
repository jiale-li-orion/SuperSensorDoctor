"""进程内事件总线 — pub/sub 模式"""

import asyncio
import fnmatch
from typing import Callable


class EventBus:
    """轻量事件总线, 支持通配符订阅"""

    def __init__(self, max_subscribers: int = 10):
        self._subscribers: list[tuple[str, Callable]] = []
        self._max = max_subscribers

    def subscribe(self, event_type: str):
        """装饰器: 订阅事件类型 (支持 * 通配符)"""
        def decorator(func: Callable):
            if len(self._subscribers) >= self._max:
                raise RuntimeError(
                    f"Max {self._max} subscribers reached"
                )
            self._subscribers.append((event_type, func))
            return func
        return decorator

    async def publish(self, event):
        """发布事件, 所有匹配的订阅者都会收到"""
        for pattern, handler in self._subscribers:
            if fnmatch.fnmatch(event.event_type, pattern):
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)

    def clear(self):
        self._subscribers.clear()
