#!/bin/bash
# h3_switch_5090.sh 的 start_vllm 决策表测试(不启动任何服务、不碰 GPU)。
# 通过 H3_SWITCH_LIB=1 source 脚本只取函数,再 stub 掉 healthy/port_busy/_launch_vllm。
# 覆盖的正是那条静默服错 checkpoint 的 bug: 健康但变体不同 -> 必须重启。
set -u
SCRIPT="${1:-$HOME/data/dropbox/CV/h3/scripts/h3_switch_5090.sh}"
RUN_REAL="$HOME/data/dropbox/CV/h3/run"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0

run_case() { # $1 desc, $2 healthy(0/1), $3 existing variant(''=none), $4 want label, $5 expect: launch|reuse
  local desc="$1" hz="$2" have="$3" want="$4" expect="$5"
  ( set +u
    export H3_SWITCH_LIB=1
    # shellcheck disable=SC1090
    source "$SCRIPT" >/dev/null 2>&1
    RUN="$TMP/run"; mkdir -p "$RUN"
    rm -f "$RUN"/vllm.*
    [ -n "$have" ] && printf '%s' "$have" > "$RUN/vllm.variant"
    [ -n "$have" ] && printf '%s' "/models/$have" > "$RUN/vllm.model"
    [ -n "$have" ] && echo 12345 > "$RUN/vllm.pid"
    healthy() { return "$hz"; }              # 0 = 健康
    port_busy() { return 1; }                # 端口空闲
    stop_one() { echo "STOPPED:$1"; rm -f "$RUN/$1.pid" "$RUN/$1.variant" "$RUN/$1.model"; }
    _launch_vllm() { echo "LAUNCHED:$2:$1"; echo 999 > "$RUN/vllm.pid"; }
    start_vllm "/models/$want" "$want"
  ) > "$TMP/out" 2>&1
  local got="reuse"; grep -q "^LAUNCHED:" "$TMP/out" && got="launch"
  if [ "$got" = "$expect" ]; then
    pass=$((pass+1)); printf '  ok   %-52s -> %s\n' "$desc" "$got"
  else
    fail=$((fail+1)); printf '  FAIL %-52s -> %s (期望 %s)\n' "$desc" "$got" "$expect"
    sed 's/^/       | /' "$TMP/out"
  fi
}

echo "start_vllm 决策表 (healthy / 已记录变体 / 请求变体 -> 期望行为)"
run_case "健康 + 同变体 base->base            = 复用"      0 base       base       reuse
run_case "健康 + 同变体 turbo->turbo          = 复用"      0 turbo-lora turbo-lora reuse
run_case "健康 + 变体不同 base->turbo         = 重启"      0 base       turbo-lora launch
run_case "健康 + 变体不同 turbo->base         = 重启"      0 turbo-lora base       launch
run_case "健康 + 无变体记录(旧状态)           = 重启"      0 ''         turbo-lora launch
run_case "不健康 + 有变体记录(半死进程)       = 清理+启动" 1 turbo-lora turbo-lora launch
run_case "不健康 + 全新                       = 启动"      1 ''         base       launch

echo
echo "端口占用保护:"
( set +u
  export H3_SWITCH_LIB=1
  source "$SCRIPT" >/dev/null 2>&1
  RUN="$TMP/run"; mkdir -p "$RUN"; rm -f "$RUN"/vllm.*
  healthy() { return 1; }
  port_busy() { return 0; }                  # 端口仍被占
  stop_one() { :; }
  _launch_vllm() { echo "LAUNCHED"; }
  start_vllm "/models/base" base
) > "$TMP/out2" 2>&1
if grep -q "REFUSED" "$TMP/out2" && ! grep -q "LAUNCHED" "$TMP/out2"; then
  pass=$((pass+1)); echo "  ok   残留占用 :8091 时拒绝启动(不撞端口)"
else
  fail=$((fail+1)); echo "  FAIL 端口占用未被拦截"; sed 's/^/       | /' "$TMP/out2"
fi

echo
echo "stop_one 清理变体文件:"
( set +u
  export H3_SWITCH_LIB=1
  source "$SCRIPT" >/dev/null 2>&1
  RUN="$TMP/run"; mkdir -p "$RUN"
  printf base > "$RUN/vllm.variant"; printf /m > "$RUN/vllm.model"
  stop_one vllm >/dev/null 2>&1
  [ ! -f "$RUN/vllm.variant" ] && [ ! -f "$RUN/vllm.model" ] && echo CLEANED
) > "$TMP/out3" 2>&1
if grep -q CLEANED "$TMP/out3"; then
  pass=$((pass+1)); echo "  ok   stop_one 同时清掉 .variant/.model(不留过期声明)"
else
  fail=$((fail+1)); echo "  FAIL stop_one 未清理变体文件"
fi

echo
echo "真实 run/ 目录未被本测试触碰: $([ -e "$RUN_REAL/vllm.variant" ] && echo '存在(未改)' || echo '不存在(符合当前 comfy 模式)')"
echo "结果: $pass 通过, $fail 失败"
[ "$fail" -eq 0 ]
