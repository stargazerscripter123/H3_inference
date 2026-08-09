#!/usr/bin/env python3
"""MiniMax H3 — 把一台机器上的所有 profile 用同一素材跑一遍并出对比表.

每个 profile 跑 warmup + N 次 timed:**只有 timed 计入结果**,warmup 用来吃掉
冷启动(切后端、权重从磁盘载入 VRAM、首单 compile/JIT)。

用法示例:
  # 5090 全部 profile,ToS 素材,各跑 warmup+1 次 timed
  scripts/h3_eval.py --first a.png --last b.png --prompt-file p.txt --name eval1

  # 只测两条 turbo 路线,各 3 次 timed 取中位,统一 NFE 6
  scripts/h3_eval.py --first a.png --last b.png --prompt-file p.txt \\
      --profiles comfy-int8-turbo-1c,vllm-fp8-turbo-tp2 --timed 3 --nfe 6

  # 看看会怎么跑,不真的跑
  scripts/h3_eval.py --first a.png --last b.png --prompt "x" --dry-run

计时口径(重要):
  * **每一遍都换 seed**(warmup=seed, timed_i=seed+i)。ComfyUI 会缓存整张 graph,
    同参数第二次提交直接返回缓存 —— 不换 seed 测到的是缓存不是生成。引擎侧无此
    缓存,换 seed 也无害,故两边统一。想要固定 seed 的纯性能对比请用 --same-seed。
  * 表里的"切换"取 warmup 那一遍的启动耗时(含 h3_switch 冷启动,分钟级);
    timed 遍的启动接近 0,因为后端已经在跑。
  * 每个 profile 的画布/默认步数取自它自己的定义(见 h3_generate.py --list),
    除非用 --width/--height 或 --nfe/--steps 显式覆盖 —— 那样才是同画布对比。

依赖: 与 h3_generate.py 相同(ffmpeg/ssh/scp + 远端已部署)。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h3_generate", _HERE / "h3_generate.py")
H3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H3)

PROJECT = H3.PROJECT
OUTPUTS = H3.OUTPUTS

# 默认评测顺序: 同后端的排在一起,少切几次(每次切换都是分钟级冷启动)
PREFERRED_ORDER = [
    "comfy-int8-original-1c",
    "comfy-int8-teacache-1c",
    "comfy-int8-turbo-1c",
    "comfy-bf16-original-4c",
    "vllm-fp8-original-tp2",
    "vllm-fp8-turbo-tp2",
    "vllm-bf16-original-tp4",
    "vllm-fp8-original-tp4",
    "vllm-bf16-original-tp2u2",
    "vllm-fp8-original-tp2u2",
    "vllm-bf16-turbo-tp4",
    "vllm-fp8-turbo-tp4",
    "vllm-bf16-turbo-tp2u2",
    "vllm-fp8-turbo-tp2u2",
    "sglang-fp8-original-tp4",
    "sglang-fp8-turbo-tp4",
]


def order_profiles(profs: list[str]) -> list[str]:
    rank = {p: i for i, p in enumerate(PREFERRED_ORDER)}
    return sorted(profs, key=lambda p: (rank.get(p, len(rank)), p))


def resolve_on_host(host: str, token: str) -> tuple[str | None, str]:
    """在某台机器上解析一个 profile 记号。

    除了 h3_generate 的全名/家族简称/旧名,这里额外接受**远端 switch 脚本自己的
    词汇**,例如 runpods 的 `turbo bf16 tp4`(或写成 turbo-bf16-tp4)—— 按 switch
    字符串精确匹配,不猜。
    """
    prof, note = H3.resolve_profile(host, token)
    if prof:
        return prof, note
    norm = " ".join(token.replace("-", " ").split())
    for p, d in H3.host_profiles(H3.HOSTS[host]).items():
        sw = d.get("switch")
        if sw and " ".join(sw.split()) == norm:
            return p, f"switch 词汇 {token!r} → {p}"
    return None, note


def with_resident(cfg: dict, prof: str, resident: int | None) -> dict:
    """把 resident 追加成 switch 的第 4 个位置参数(只对声明支持的机器)。"""
    if resident is None:
        return cfg
    sv = dict(H3.host_profiles(cfg)[prof])
    sv["switch"] = f"{sv['switch']} {resident}"
    out = dict(cfg)
    out["serving"] = {**cfg.get("serving", {}), prof: sv}
    return out


def make_ns(a: argparse.Namespace, profile: str, seed: int,
            canvas: tuple[int, int]) -> argparse.Namespace:
    """构造 h3_generate.run_host 需要的 Namespace(它只读这几个字段)。"""
    ns = argparse.Namespace()
    ns.profile = profile
    ns.seed = seed
    ns.seconds = a.seconds
    ns.nfe = a.nfe
    ns.steps = a.steps
    ns.steps_explicit = a.steps is not None
    ns.canvas_w, ns.canvas_h = canvas
    ns.length = a.length
    return ns


def prepare_workdir(a: argparse.Namespace, root: Path, canvas: tuple[int, int],
                    prompt_text: str) -> Path:
    """每一遍都要有自己的工作目录(run_host 从这里上传、也把结果放回这里)。"""
    root.mkdir(parents=True, exist_ok=True)
    cw, ch = canvas
    H3.preprocess(a.first, root / "first.png", cw, ch)
    H3.preprocess(a.last, root / "last.png", cw, ch)
    (root / "prompt.txt").write_text(prompt_text)
    return root


def run_once(tag: str, cfg: dict, profile: str, seed: int, canvas: tuple[int, int],
             a: argparse.Namespace, job: str, workdir: Path,
             prompt_text: str) -> dict:
    prepare_workdir(a, workdir, canvas, prompt_text)
    ns = make_ns(a, profile, seed, canvas)
    results: dict = {}
    t0 = time.time()
    H3.run_host(tag, cfg, job, workdir, ns, results)
    status, detail, *extra = results.get(tag, ("FAILED", "no result"))
    timing = extra[0] if extra and isinstance(extra[0], dict) else {}
    return {
        "ok": status == "OK",
        "detail": detail,
        "seed": seed,
        "wall_total_s": round(time.time() - t0, 1),
        "startup_s": timing.get("startup_s"),
        "infer_s": timing.get("infer_s"),
        "peak_vram": timing.get("peak_vram"),
    }


def fmt_steps(pdef: dict, a: argparse.Namespace) -> str:
    if a.nfe is not None:
        return f"NFE {a.nfe}"
    if a.steps is not None:
        return f"{a.steps} 步"
    if pdef.get("nfe"):
        return f"NFE {pdef['nfe']}"
    return f"{pdef.get('steps', 30)} 步"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", required=True, type=Path, help="首帧图片(本机路径)")
    ap.add_argument("--last", required=True, type=Path, help="尾帧图片(本机路径)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="prompt 文本")
    g.add_argument("--prompt-file", type=Path, help="prompt 文本文件")
    ap.add_argument("--host", default="5090,runpods",
                    help="被测机器,逗号分隔(默认 5090,runpods);可选: "
                         + "|".join(H3.HOSTS) + ";all=全部")
    ap.add_argument("--profiles", default="all",
                    help="逗号分隔的 profile 列表;all=各机器全部。除全名外还接受家族简称、"
                         "旧名、以及远端 switch 自己的词汇(如 runpods 的 "
                         "'turbo bf16 tp4')。同一记号在不同机器上各自解析,"
                         "机器上没有的自动跳过")
    ap.add_argument("--resident", default=None,
                    help="DLO 常驻层数,逗号分隔可扫多档(如 40,50)。只对声明支持的机器"
                         "生效(目前 runpods);不给则用该机 switch 的默认值。"
                         "它只影响显存/调度不改变输出,故不进 profile 名")
    ap.add_argument("--timed", type=int, default=1,
                    help="每个 profile 计入结果的遍数(默认 1,即总共跑 2 遍)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0, help="warmup 用它,timed 用 seed+1,+2…")
    ap.add_argument("--same-seed", action="store_true",
                    help="所有遍用同一 seed。仅在纯引擎 profile 上安全 —— ComfyUI "
                         "会缓存整图,这样会把缓存当成生成速度测")
    ap.add_argument("--nfe", type=int, default=None,
                    help="统一 NFE(不给则每个 profile 用自己的默认值)")
    ap.add_argument("--steps", type=int, default=None, help="统一步数(与 --nfe 二选一)")
    ap.add_argument("--width", type=int, default=None,
                    help="统一画布(不给则每个 profile 用自己的默认画布)")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--name", default=None, help="评测名(默认时间戳)")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划,不执行")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="某个 profile 失败就中止(默认继续测其余的)")
    a = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe", "ssh", "scp"):
        if not shutil.which(tool):
            print(f"缺少 {tool}", file=sys.stderr)
            return 1
    for f in (a.first, a.last):
        if not f.exists():
            print(f"文件不存在: {f}", file=sys.stderr)
            return 1
    if a.timed < 1:
        print("--timed 至少为 1", file=sys.stderr)
        return 1

    hosts = list(H3.HOSTS) if a.host == "all" else [
        t.strip() for t in a.host.split(",") if t.strip()]
    for t in hosts:
        if t not in H3.HOSTS:
            print(f"未知机器 {t};可选: {', '.join(H3.HOSTS)}", file=sys.stderr)
            return 1

    residents: list[int | None] = [None]
    if a.resident:
        try:
            residents = [int(x) for x in a.resident.split(",") if x.strip()]
        except ValueError:
            print("--resident 只能是整数列表,如 40,50", file=sys.stderr)
            return 1

    # 逐机解析 profile —— 同一记号在不同机器上可解析成不同全名,机器上没有的跳过
    plan: list[tuple[str, str, int | None]] = []
    for t in hosts:
        supports_res = "resident_arg" in H3.HOSTS[t]
        res_list = residents if supports_res else [None]
        if a.resident and not supports_res:
            print(f"[note] {t}: 该机 switch 不接 resident 参数,忽略 --resident")
        avail_t = H3.host_profiles(H3.HOSTS[t])
        if a.profiles == "all":
            got = order_profiles(list(avail_t))
        else:
            got = []
            for want in [p.strip() for p in a.profiles.split(",") if p.strip()]:
                prof, note = resolve_on_host(t, want)
                if prof:
                    got.append(prof)
                    if note:
                        print(f"[note] {t}: {note}")
                else:
                    print(f"[skip] {t}: {note}", file=sys.stderr)
            got = order_profiles(list(dict.fromkeys(got)))
        for p in got:
            # resident 只对 serving profile 有意义(ComfyUI 那边没有 DLO)
            rl = res_list if avail_t[p]["_kind"] == "serving" else [None]
            plan += [(t, p, r) for r in rl]
    if not plan:
        print("没有可测的 profile;用 h3_generate.py --list 看能力矩阵", file=sys.stderr)
        return 1

    a.length = H3.snap_length(a.seconds)
    prompt_text = a.prompt_file.read_text() if a.prompt_file else a.prompt
    name = a.name or time.strftime("eval_%m%d_%H%M%S")
    name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    root = OUTPUTS / name

    # 画布: 显式覆盖 > 各 profile 自己的定义(按首帧方向转竖版)
    w, h = H3.probe_size(a.first)
    def pdef_of(t: str, p: str) -> dict:
        return H3.host_profiles(H3.HOSTS[t])[p]

    def canvas_for(t: str, p: str) -> tuple[int, int]:
        if a.width and a.height:
            return (a.width, a.height)
        bw, bh = pdef_of(t, p).get("canvas", (832, 480))
        return (bw, bh) if w >= h else (bh, bw)

    def canvas_label(t: str, p: str) -> str:
        # SGLang 的 API 硬锁短边 768,传什么画布都不生效 —— 别让表里的数字骗人
        if pdef_of(t, p).get("client") == "run_fl2va_sglang.py":
            return "1376×768(API 锁)"
        cw, chh = canvas_for(t, p)
        return f"{cw}×{chh}"

    print(f"评测 {name} @ " + ", ".join(f"{t}({H3.HOSTS[t]['ssh']})" for t in hosts))
    print(f"  {len(plan)} 个 (机器,profile) 组合 × (1 warmup + {a.timed} timed), "
          f"{a.length} 帧 (~{a.length/24:.2f}s), seed {a.seed}"
          f"{'(全程同一 seed)' if a.same_seed else '(逐遍 +1 以破 ComfyUI 整图缓存)'}")
    for t, p, r in plan:
        res = f"  r{r}" if r is not None else ""
        print(f"    {t:<8} {p:<26} {H3.profile_backend(H3.HOSTS[t], p):<24} "
              f"{fmt_steps(pdef_of(t, p), a):<8} {canvas_label(t, p)}{res}")
    est = len(plan) * (1 + a.timed)
    print(f"  共 {est} 次生成;每换一个后端都有分钟级冷启动,请预留时间。")
    if a.dry_run:
        print("\n(--dry-run,未执行)")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "name": name,
        "hosts": {t: H3.HOSTS[t]["ssh"] for t in hosts},
        "material": {"first": str(a.first), "last": str(a.last),
                     "seconds": a.seconds, "frames": a.length,
                     "prompt_chars": len(prompt_text)},
        "protocol": {"warmup": 1, "timed": a.timed,
                     "seed_base": a.seed, "same_seed": a.same_seed,
                     "note": "只有 timed 计入;warmup 吃掉切换与冷加载"},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profiles": {},   # key = "<host>/<profile>"
    }

    failed_any = False
    for i, (tag, prof, resident) in enumerate(plan, 1):
        cfg = with_resident(H3.HOSTS[tag], prof, resident)
        canvas = canvas_for(tag, prof)
        rsfx = f"@r{resident}" if resident is not None else ""
        key = f"{tag}/{prof}{rsfx}"
        print(f"\n{'='*72}\n[{i}/{len(plan)}] {key}\n{'='*72}", flush=True)
        entry = {
            "host": tag,
            "profile": prof,
            "backend": H3.profile_backend(cfg, prof),
            "canvas": canvas_label(tag, prof),
            "steps": fmt_steps(pdef_of(tag, prof), a),
            "resident": resident,
            "switch": H3.host_profiles(cfg)[prof].get("switch"),
            "warmup": None, "timed": [],
        }
        slug = f"{tag}_{prof}" + (f"_r{resident}" if resident is not None else "")

        wu_seed = a.seed
        print(f"--- warmup (seed {wu_seed}, 不计入) ---", flush=True)
        wu = run_once(tag, cfg, prof, wu_seed, canvas, a,
                      f"{name}_{slug}_warmup", root / slug / "warmup", prompt_text)
        entry["warmup"] = wu
        if not wu["ok"]:
            print(f"!! warmup 失败: {wu['detail']}")
            entry["error"] = f"warmup failed: {wu['detail']}"
            report["profiles"][key] = entry
            failed_any = True
            (root / "eval_results.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2))
            if a.stop_on_fail:
                break
            continue

        for k in range(1, a.timed + 1):
            seed = a.seed if a.same_seed else a.seed + k
            print(f"--- timed {k}/{a.timed} (seed {seed}) ---", flush=True)
            r = run_once(tag, cfg, prof, seed, canvas, a,
                         f"{name}_{slug}_timed{k}", root / slug / f"timed{k}",
                         prompt_text)
            entry["timed"].append(r)
            if not r["ok"]:
                print(f"!! timed{k} 失败: {r['detail']}")
                failed_any = True
                if a.stop_on_fail:
                    break

        good = [r["infer_s"] for r in entry["timed"] if r["ok"] and r["infer_s"]]
        if good:
            entry["infer_median_s"] = round(statistics.median(good), 1)
            entry["infer_all_s"] = good
            entry["switch_s"] = wu.get("startup_s")
            entry["peak_vram"] = next(
                (r["peak_vram"] for r in entry["timed"] if r.get("peak_vram")), None)
            print(f"==> {key}: 推理中位 {entry['infer_median_s']}s "
                  f"(各次 {good}), 切换 {entry['switch_s']}s")
        report["profiles"][key] = entry
        (root / "eval_results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2))

    # ---- 汇总表 ----
    rows = []
    for _key, e in report["profiles"].items():
        res = str(e.get("resident")) if e.get("resident") is not None else "默认"
        if "infer_median_s" not in e:
            rows.append((e["host"], e["profile"], res, e["steps"],
                         e["canvas"], "—", "—", "—", e.get("error", "失败")))
            continue
        rows.append((e["host"], e["profile"], res, e["steps"], e["canvas"],
                     f"{e['infer_median_s']}s",
                     "/".join(str(x) for x in e["infer_all_s"]),
                     f"{e['switch_s']}s", e.get("peak_vram") or "—"))

    md = [f"# H3 profile 评测 · {'+'.join(hosts)} · {name}", "",
          f"素材 {a.first.name} → {a.last.name},{a.length} 帧 (~{a.length/24:.2f}s);"
          f" 协议 = 1 warmup(不计) + {a.timed} timed"
          f"{'(同 seed)' if a.same_seed else '(逐遍换 seed 破缓存)'}", "",
          "| 机器 | profile | resident | 步数 | 画布 | 推理(中位) | 各次 | 切换 | 峰值显存 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md += ["", "注:「切换」取 warmup 那一遍的启动耗时(含 h3_switch 冷启动);",
           "timed 遍的启动接近 0,不重复计入。产物见各 profile 子目录。"]
    table = "\n".join(md)
    (root / "REPORT.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\n输出目录: {root}")
    print(f"  eval_results.json / REPORT.md / <profile>/{{warmup,timed*}}/")
    return 2 if failed_any else 0


if __name__ == "__main__":
    sys.exit(main())
