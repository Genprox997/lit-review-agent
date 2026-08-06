"""离线测试：LLM 批量并发（P3-4）。

stub 模式走串行短路；真实 provider 走线程池并发。这里强制并发分支并替换
底层 chat，验证（1）结果顺序与输入严格对应、（2）并发确实缩短端到端时延。
"""

from __future__ import annotations

import time

from src.agent import llm
from src.config import Settings


def test_chat_json_many_is_ordered(monkeypatch):
    s = Settings()
    s.llm_provider = "deepseek"  # 强制走并发分支（非 stub）
    s.llm_max_workers = 4
    monkeypatch.setattr(llm, "get_settings", lambda: s)

    def fake_chat(node, system, user, *, json_mode=False, settings=None):
        idx = int(user.split("CASE ")[1])
        return '{"case": %d}' % idx

    monkeypatch.setattr(llm, "chat", fake_chat)

    items = [{"node": "c", "system": "s", "user": f"CASE {i}"} for i in range(4)]
    res = llm.chat_json_many(items, default={})
    assert [r["case"] for r in res] == [0, 1, 2, 3]


def test_chat_json_many_runs_concurrently(monkeypatch):
    s = Settings()
    s.llm_provider = "deepseek"
    s.llm_max_workers = 4
    monkeypatch.setattr(llm, "get_settings", lambda: s)

    def fake_chat(node, system, user, *, json_mode=False, settings=None):
        time.sleep(0.2)  # 单条 0.2s；4 条并发应远小于 0.8s
        idx = int(user.split("CASE ")[1])
        return '{"case": %d}' % idx

    monkeypatch.setattr(llm, "chat", fake_chat)

    items = [{"node": "c", "system": "s", "user": f"CASE {i}"} for i in range(4)]
    t0 = time.time()
    res = llm.chat_json_many(items, default={})
    dt = time.time() - t0
    assert [r["case"] for r in res] == [0, 1, 2, 3]
    assert dt < 0.6, f"期望并发 <0.6s，实际 {dt:.2f}s"


def test_chat_many_preserves_order(monkeypatch):
    s = Settings()
    s.llm_provider = "deepseek"
    s.llm_max_workers = 3
    monkeypatch.setattr(llm, "get_settings", lambda: s)

    def fake_chat(node, system, user, *, json_mode=False, settings=None):
        idx = int(user.split("CASE ")[1])
        return f"text-{idx}"

    monkeypatch.setattr(llm, "chat", fake_chat)

    items = [{"node": "c", "system": "s", "user": f"CASE {i}"} for i in range(3)]
    res = llm.chat_many(items)
    assert res == ["text-0", "text-1", "text-2"]


def test_single_item_short_circuits_serial(monkeypatch):
    """单条时不应起线程池，直接返回。"""
    s = Settings()
    s.llm_provider = "deepseek"
    monkeypatch.setattr(llm, "get_settings", lambda: s)

    calls = []

    def fake_chat(node, system, user, *, json_mode=False, settings=None):
        calls.append(user)
        return '{"ok": 1}'

    monkeypatch.setattr(llm, "chat", fake_chat)
    res = llm.chat_json_many([{"node": "c", "system": "s", "user": "CASE 0"}], default={})
    assert res == [{"ok": 1}]
    assert len(calls) == 1
