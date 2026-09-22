"""HTTP 接口测试。

覆盖：
* 峰值点接口返回的立方窗口与最终采用值精确等于峰值；
* 越过/未越过峰值的大小关系经过 HTTP 往返后仍成立；
* 低带宽时延积下 TCP 友好分支被选中，高 BDP 下走 CUBIC；
* 四类非法输入各自返回不同 error code 的结构化 422（含 w_max=0）；
* 单点与轨迹接口在同一时刻结果完全一致；
* 快速收敛字段随峰值升降变化；
* 并发多请求互不串扰；
* 标准情形只读接口与健康检查。
"""

from __future__ import annotations

import concurrent.futures

import pytest

from app.window import kappa


def _eval(client, **payload):  # type: ignore[no-untyped-def]
    return client.post("/api/v3/evaluate", json=payload)


# ---------- 峰值点与曲线方向 ----------


def test_http_peak_point_is_exactly_wmax(client) -> None:  # type: ignore[no-untyped-def]
    w_max = 100.0
    k = kappa(w_max)
    resp = _eval(client, w_max=w_max, rtt=1.0, t=k)
    assert resp.status_code == 200
    body = resp.json()
    assert body["w_cubic"] == w_max
    assert body["w"] == w_max
    assert body["at_peak"] is True
    assert body["k"] == k


def test_http_above_and_below_peak(client) -> None:  # type: ignore[no-untyped-def]
    w_max = 200.0
    k = kappa(w_max)
    below = _eval(client, w_max=w_max, rtt=1.0, t=0.5 * k).json()
    above = _eval(client, w_max=w_max, rtt=1.0, t=1.5 * k).json()
    assert below["w_cubic"] < w_max < above["w_cubic"]
    assert below["at_peak"] is False and above["at_peak"] is False


def test_http_larger_wmax_postpones_return_to_peak(client) -> None:  # type: ignore[no-untyped-def]
    k1 = _eval(client, w_max=100, rtt=1, t=0).json()["k"]
    k2 = _eval(client, w_max=400, rtt=1, t=0).json()["k"]
    assert k2 > k1


def test_http_doubling_c_shortens_return_time(client) -> None:  # type: ignore[no-untyped-def]
    k1 = _eval(client, w_max=100, rtt=1, t=0, c=0.4).json()["k"]
    k2 = _eval(client, w_max=100, rtt=1, t=0, c=0.8).json()["k"]
    assert k2 < k1


# ---------- 分支选择 ----------


def test_low_bdp_selects_tcp_friendly_branch(client) -> None:  # type: ignore[no-untyped-def]
    # 小 RTT 使线性 Reno 增长远快于立方初期增长 → 退让到 TCP 友好区。
    w_max = 10.0
    t = kappa(w_max) / 2.0
    body = _eval(client, w_max=w_max, rtt=0.01, t=t).json()
    assert body["branch"] == "tcp_friendly"
    assert body["tcp_friendly"] is True
    assert body["w"] == body["w_tcp"]
    assert body["w_tcp"] > body["w_cubic"]


def test_high_bdp_selects_cubic_branch(client) -> None:  # type: ignore[no-untyped-def]
    w_max = 100.0
    body = _eval(client, w_max=w_max, rtt=1.0, t=0.5 * kappa(w_max)).json()
    assert body["branch"] == "cubic"
    assert body["tcp_friendly"] is False
    assert body["w"] == body["w_cubic"]
    assert body["w_cubic"] >= body["w_tcp"]


# ---------- 非法输入：四类可区分错误 ----------


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"w_max": -3, "rtt": 0.1, "t": 1.0}, "invalid_wmax"),
        ({"w_max": 0, "rtt": 0.1, "t": 1.0}, "invalid_wmax"),
        ({"w_max": 10, "rtt": 0.1, "t": 1.0, "c": -1}, "invalid_c"),
        ({"w_max": 10, "rtt": 0.1, "t": 1.0, "c": 0}, "invalid_c"),
        ({"w_max": 10, "rtt": 0.1, "t": -0.5}, "invalid_time"),
        ({"w_max": 10, "rtt": 0, "t": 1.0}, "invalid_rtt"),
        ({"w_max": 10, "rtt": -0.2, "t": 1.0}, "invalid_rtt"),
    ],
)
def test_invalid_inputs_return_typed_errors(client, payload, code) -> None:  # type: ignore[no-untyped-def]
    resp = _eval(client, **payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert "field" in body["error"]


def test_non_finite_input_is_rejected(client) -> None:  # type: ignore[no-untyped-def]
    # 用裸 JSON 体发送 NaN 标记（httpx 的 json= 参数会在客户端拒绝），
    # 由服务端校验挡下并给出可区分错误码。
    resp = client.post(
        "/api/v3/evaluate",
        content='{"w_max": NaN, "rtt": 0.1, "t": 1.0}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "non_finite_number"

    resp_inf = client.post(
        "/api/v3/evaluate",
        content='{"w_max": 10, "rtt": 0.1, "t": Infinity}',
        headers={"content-type": "application/json"},
    )
    assert resp_inf.status_code == 422
    assert resp_inf.json()["error"]["code"] == "non_finite_number"


def test_wmax_zero_is_illegal_not_zero_window(client) -> None:  # type: ignore[no-untyped-def]
    resp = _eval(client, w_max=0, rtt=0.1, t=1.0)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_wmax"


def test_errors_never_escape_as_500(client) -> None:  # type: ignore[no-untyped-def]
    resp = _eval(client, w_max="abc", rtt=0.1, t=1.0)
    assert resp.status_code == 422
    assert "error" in resp.json()


# ---------- 单点 / 轨迹 一致性 ----------


def test_trajectory_matches_single_point_for_every_sample(client) -> None:  # type: ignore[no-untyped-def]
    w_max, rtt = 120.0, 0.2
    k = kappa(w_max)
    times = [0.0, 0.25 * k, 0.5 * k, k, 1.25 * k, 2.0 * k]
    traj = client.post(
        "/api/v3/trajectory",
        json={"w_max": w_max, "rtt": rtt, "times": times},
    )
    assert traj.status_code == 200
    traj_body = traj.json()
    assert traj_body["k"] == k

    fields = ("w_cubic", "w_tcp", "w", "branch", "tcp_friendly", "at_peak")
    for t, sample in zip(times, traj_body["samples"]):
        single = _eval(client, w_max=w_max, rtt=rtt, t=t).json()
        assert sample["t"] == t
        for field in fields:
            assert sample[field] == single[field], (t, field)
        assert {"w_cubic", "w_tcp", "w", "branch"} <= set(sample)


def test_trajectory_peak_sample_exactly_wmax(client) -> None:  # type: ignore[no-untyped-def]
    w_max = 64.0
    k = kappa(w_max)
    body = client.post(
        "/api/v3/trajectory",
        json={"w_max": w_max, "rtt": 1.0, "times": [0.0, k]},
    ).json()
    assert body["samples"][1]["w_cubic"] == w_max


@pytest.mark.parametrize(
    ("w_max", "rtt", "c"),
    [
        # 高 BDP：立方分支主导；系数取小/大一倍以上/非标值
        (100.0, 1.0, 0.1),
        (100.0, 1.0, 0.8),
        (120.0, 0.2, 2.5),
        # 低 BDP：非默认系数下线性分支仍可能接管，分支也必须逐点一致
        (10.0, 0.01, 0.8),
        (10.0, 0.01, 1.7),
    ],
)
def test_trajectory_matches_single_point_for_non_default_c(
    client, w_max, rtt, c  # type: ignore[no-untyped-def]
) -> None:
    """非默认立方系数下，轨迹每个采样点必须与单点接口逐字段完全一致。

    回归用例：轨迹顶层 K 用的是请求里的 C，而逐点求值若偷偷落回
    默认 C=0.4，则同一时刻两接口的窗口/分支会对不上，且 t=K 采样点
    不会落在峰值。这里连同顶层 K 与峰值锚定一起卡死。
    """
    assert c != 0.4  # 本用例只允许非默认系数
    k = kappa(w_max, c)
    times = [0.0, 0.25 * k, 0.5 * k, k, 1.25 * k, 2.0 * k]

    traj = client.post(
        "/api/v3/trajectory",
        json={"w_max": w_max, "rtt": rtt, "c": c, "times": times},
    )
    assert traj.status_code == 200
    traj_body = traj.json()

    # 顶层报的回到峰值时间必须就是传入系数对应的 K
    assert traj_body["c"] == c
    assert traj_body["k"] == k

    fields = ("w_cubic", "w_tcp", "w", "branch", "tcp_friendly", "at_peak")
    for t, sample in zip(times, traj_body["samples"]):
        single = _eval(client, w_max=w_max, rtt=rtt, t=t, c=c).json()
        assert sample["t"] == t
        for field in fields:
            assert sample[field] == single[field], (c, t, field)

    # 用传入系数算出的 K 时刻，轨迹采样点必须精确回到峰值并标记 at_peak
    peak_sample = traj_body["samples"][times.index(k)]
    assert peak_sample["at_peak"] is True
    assert peak_sample["w_cubic"] == w_max


def test_trajectory_requires_increasing_times(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.post(
        "/api/v3/trajectory",
        json={"w_max": 10, "rtt": 0.1, "times": [0.0, 1.0, 1.0, 2.0]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "non_increasing_times"


def test_trajectory_rejects_negative_time(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.post(
        "/api/v3/trajectory",
        json={"w_max": 10, "rtt": 0.1, "times": [0.0, -1.0]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_time"


# ---------- 快速收敛 ----------


def test_fast_convergence_flag_toggles(client) -> None:  # type: ignore[no-untyped-def]
    down = _eval(client, w_max=100, rtt=1, t=1, w_last_max=140).json()
    up = _eval(client, w_max=100, rtt=1, t=1, w_last_max=80).json()
    first = _eval(client, w_max=100, rtt=1, t=1).json()
    assert down["fast_convergence"] is True
    assert up["fast_convergence"] is False
    assert first["fast_convergence"] is False


# ---------- 并发互不串扰 ----------


def test_concurrent_requests_do_not_cross_talk(client) -> None:  # type: ignore[no-untyped-def]
    """并发发出参数互不相同的请求，每个响应必须只反映自己的入参。"""
    params = [
        (10.0, 0.01, 0.5),
        (50.0, 0.05, 2.0),
        (100.0, 1.0, 0.0),
        (100.0, 1.0, kappa(100.0)),
        (300.0, 0.2, 1.5),
        (7.0, 0.002, 0.3),
        (80.0, 0.5, 3.3),
        (250.0, 0.08, 0.9),
    ]

    def hit(args: tuple[float, float, float]) -> dict:
        w_max, rtt, t = args
        resp = _eval(client, w_max=w_max, rtt=rtt, t=t)
        assert resp.status_code == 200
        return resp.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda p: [hit(p) for _ in range(25)], params))

    for (w_max, rtt, t), rounds in zip(params, results):
        for body in rounds:
            assert body["w_max"] == w_max
            assert body["rtt"] == rtt
            assert body["t"] == t
            assert body["w_cubic"] > 0
            assert body["branch"] in ("cubic", "tcp_friendly")
            # 每个响应的 K 只由它自己的 W_max/C 决定
            assert body["k"] == pytest.approx(kappa(w_max), rel=1e-12)


def test_trajectory_concurrent_isolation(client) -> None:  # type: ignore[no-untyped-def]
    specs = [
        (10.0, [0.0, 0.1, 0.2]),
        (100.0, [0.0, 1.0, 2.0, 3.0]),
        (50.0, [0.0, 0.5, kappa(50.0)]),
    ]

    def run(spec):  # type: ignore[no-untyped-def]
        w_max, times = spec
        resp = client.post(
            "/api/v3/trajectory", json={"w_max": w_max, "rtt": 0.1, "times": times}
        )
        return w_max, times, resp

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(run, specs * 20))

    for w_max, times, resp in outs:
        assert resp.status_code == 200
        body = resp.json()
        assert body["w_max"] == w_max
        assert [s["t"] for s in body["samples"]] == times


# ---------- 标准情形 / 监控 ----------


def test_standards_readonly_echo_preset(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/api/v3/standards")
    assert resp.status_code == 200
    body = resp.json()
    assert body["constants"]["beta_multiplicative_decrease"] == 0.7
    assert body["constants"]["default_c"] == 0.4

    preset = body["preset_example"]
    inputs_ = preset["inputs"]
    w_max = inputs_["w_max"]
    k = preset["k"]

    # 起点：低于峰值
    origin = preset["samples"][0]
    assert origin["t"] == 0.0
    assert origin["w_cubic"] == pytest.approx(0.7 * w_max)
    assert origin["w_cubic"] < w_max

    # 回到峰值：精确等于
    peak_sample = next(s for s in preset["samples"] if s["at_peak"])
    assert peak_sample["w_cubic"] == w_max
    assert preset["anchors"]["return_to_peak"]["expected_w_cubic"] == w_max
    assert preset["anchors"]["return_to_peak"]["t"] == k

    # 越过峰值：严格大于
    beyond = preset["samples"][-1]
    assert beyond["t"] == 2.0 * k
    assert beyond["w_cubic"] > w_max

    # 预置算例里上次峰值更高 → 快速收敛为开
    assert preset["fast_convergence"] is True

    # 每个采样点都带齐四要素
    for sample in preset["samples"]:
        assert {"w_cubic", "w_tcp", "w", "branch"} <= set(sample)


def test_standards_is_stable_across_reads(client) -> None:  # type: ignore[no-untyped-def]
    first = client.get("/api/v3/standards").json()
    second = client.get("/api/v3/standards").json()
    assert first == second


def test_healthz_reports_monitoring_state(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["stateless"] is True
    assert isinstance(body["requests_completed"], int)
    assert body["requests_completed"] >= 1


def test_root_lists_endpoints(client) -> None:  # type: ignore[no-untyped-def]
    body = client.get("/").json()
    assert any("/api/v3/evaluate" in e for e in body["endpoints"])
