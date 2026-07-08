"""数据回放引擎 — 生成合成数据"""

from datetime import datetime, timedelta
from typing import Generator
from agent_layer.state_objects import StateObject


class ReplayEngine:
    """合成数据生成器, 模拟场景"""

    @staticmethod
    def generate_synthetic(duration_sec: int = 30) -> Generator[StateObject, None, None]:
        """生成合成数据: 指定时长(秒), 每秒一个 StateObject"""
        import random
        rng = random.Random(42)
        start = datetime.now()

        hr = 72.0
        rr = 16.0
        temp = 36.5

        for i in range(duration_sec):
            ts = start + timedelta(seconds=i)

            # 随机波动
            hr += rng.gauss(0, 2)
            rr += rng.gauss(0, 0.5)
            temp += rng.gauss(0, 0.05)

            # 注入异常模式
            if 15 <= i <= 25:
                hr += 3  # 心率轻度升高
            if 30 <= i <= 35:
                hr += 25  # 心率显著升高 (模拟异常)
            if 40 <= i <= 45:
                temp += 1.5  # 体温升高

            yield StateObject(
                window_id=f"syn_{i:04d}",
                timestamp=ts,  # datetime 对象, 与 StateObject.timestamp: datetime 类型一致
                heart_rate=max(40, min(180, round(hr, 1))),
                respiration_rate=max(8, min(30, round(rr, 1))),
                body_temp=round(temp, 1),
                wifi_confidence=0.9,
                mmwave_confidence=0.85,
                fall_status="fall" if 30 <= i <= 35 else "no_fall",
                activity_state="rest",
                source="synthetic",
            )
