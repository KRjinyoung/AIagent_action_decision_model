"""
G2(plan_task / ask_user / web_search) 전문가 — 전처리(자체완결).

raw 레코드(dict; session_meta/history/current_prompt) -> 피처 DataFrame.
구성: (1) ext 구조피처 68 (exp04 build_stage2_features 원본 복제)
      (2) rich 프롬프트/결과피처 36 (exp11 build_rich_feats 원본 복제)
      (3) 타겟피처 7 (G2 오답분석 log-odds: 에러/막힘=ask, 단계+첫턴=plan, 공식/최신+탐색뒤=web)
      (4) tier 상대 z-score 2 (budget/loc, train 통계로 fit) + src_au
학습(train)과 추론(predict)이 반드시 이 파일을 글자 그대로 공유한다(train/serve skew 방지).
"""
import json, os, re
import numpy as np
import pandas as pd

ACTIONS = ["read_file","grep_search","list_directory","glob_pattern","edit_file",
           "write_file","apply_patch","run_bash","run_tests","lint_or_typecheck",
           "ask_user","plan_task","web_search","respond_only"]
EXPLORE_A = ["read_file","grep_search","list_directory","glob_pattern"]
EDIT_A = {"write_file","edit_file","apply_patch"}
EXEC_A = {"run_bash","run_tests","lint_or_typecheck"}
TOP_LANGS = ["py","ts","tsx","js","jsx","rs","java","go","vue","yaml","sql","md","c","cpp","rb"]

CAT_COLS = ["prev_action","prev_action2","last_ci","user_tier","language_pref"]
EXT_NUM = (["turn_index","elapsed","budget","loc","git_dirty","n_hist","n_open","n_lang",
            "first_turn","open_empty","open_n_ext","open_has_deep","dom_ratio",
            "prompt_ref_open","n_distinct_act","n_explore_done"]
           + [f"oext_{L}" for L in TOP_LANGS] + [f"lm_{L}" for L in TOP_LANGS]
           + [f"c_{a}" for a in ACTIONS])
RICH_COLS = ["kw_test","kw_lint","kw_runbuild","kw_web","kw_plan","kw_ask","kw_read","kw_list","kw_edit",
             "p_q","p_uncert","p_2nd","p_len","p_words","p_haspath","ref_openfile","ref_hist_path",
             "ref_hist_token","n_shared_hist_tok","last_r_pass","last_r_matches","last_r_files",
             "last_r_ladd","last_r_ldel","last_r_tests","last_r_readlines","last_is_edit","last_is_exec",
             "last_is_explore","r3_explore","r3_edit","r3_exec","n_since_edit","n_since_exec","n_user_turns"]
TGT_COLS = ["f_err","f_stuck","f_ask_any","f_plan","f_web","f_plan_first","f_web_invest"]
NEWF_COLS = ["src_au","budget_rel_tier_z","loc_rel_tier_z"]

# ---- ext: exp04 build_stage2_features 원본 복제 ----
def build_ext(sample):
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    hist = sample.get("history") or []
    prompt = sample.get("current_prompt", "") or ""
    acts = [h for h in hist if h.get("role") == "assistant_action"]
    names = [a.get("name") for a in acts]
    of = ws.get("open_files", []) or []
    exts = [os.path.splitext(p)[1].lstrip(".").lower() for p in of if "." in os.path.basename(p)]
    lm = ws.get("language_mix", {}) or {}
    es = set(exts)
    bns = [os.path.basename(p) for p in of]
    d = {
        "prev_action": names[-1] if names else "none",
        "prev_action2": names[-2] if len(names) >= 2 else "none",
        "last_ci": ws.get("last_ci_status", "none"),
        "user_tier": sm.get("user_tier", "none"),
        "language_pref": sm.get("language_pref", "none"),
        "turn_index": sm.get("turn_index", 0) or 0,
        "elapsed": sm.get("elapsed_session_sec", 0) or 0,
        "budget": sm.get("budget_tokens_remaining", 0) or 0,
        "loc": ws.get("loc", 0) or 0,
        "git_dirty": int(bool(ws.get("git_dirty"))),
        "n_hist": len(hist),
        "n_open": len(of),
        "n_lang": len(lm),
        "first_turn": int(len(acts) == 0),
        "open_empty": int(len(of) == 0),
        "open_n_ext": len(es),
        "open_has_deep": int(any(p.count("/") >= 2 for p in of)),
        "dom_ratio": max(lm.values()) if lm else 0.0,
        "prompt_ref_open": int(any(b and b in prompt for b in bns)),
        "n_distinct_act": len(set(names)),
        "n_explore_done": sum(names.count(x) for x in EXPLORE_A),
    }
    for L in TOP_LANGS:
        d[f"oext_{L}"] = int(L in es)
        d[f"lm_{L}"] = float(lm.get(L, 0.0))
    for a in ACTIONS:
        d[f"c_{a}"] = names.count(a)
    return d

# ---- rich: exp11 build_rich_feats 원본 복제 ----
_KW = {
 'kw_test': r'\btest|테스트|spec|스펙|돌려|run the (unit|profile|integration)?\s*test',
 'kw_lint': r'lint|정적\s*분석|타입\s*체크|typecheck|type-check|포맷|format|mypy|flake|eslint|린트',
 'kw_runbuild': r'\brun\b|build|빌드|실행|rebuild|재빌드|compile|컴파일|docker|make\b',
 'kw_web': r'search|검색|찾아|google|구글|docs?|문서|공식|latest|최신|버전|version|changelog|stack\s*overflow|찾아봐',
 'kw_plan': r'plan|계획|접근|approach|설계|design|단계|step|어떻게|방법|전략|strategy|나눠|쪼개|break\s*down|어떤\s*순서',
 'kw_ask': r'골라|정해|어느\s*(쪽|것)|맞아\?|decide|choose|which|should i|prefer|취향|알아서|괜찮을까|괜찮아\?|해도\s*돼',
 'kw_read': r'read|열어|보여|show me|보자|봐줘|impl|구현|내용|current impl|한번 봐',
 'kw_list': r'list|디렉토리|뭐\s*있|폴더|folder|ls\b|structure|구조|들어\s*있',
 'kw_edit': r'수정|고쳐|추가|바꿔|fix|change|add|patch|리팩|refactor|rename|삭제|remove|drop in|touch',
}
_KW = {k: re.compile(v, re.I) for k, v in _KW.items()}
_Q = re.compile(r'\?')
_UNCERT = re.compile(r'모르|헷갈|not sure|maybe|아마|어느\s*게|긴가민가|잘\s*몰라|unsure', re.I)
_SECOND = re.compile(r'\byou\b|너가|네가|알아서|해줘|해봐|줄래|can you|could you', re.I)
_PATHTOK = re.compile(r'[\w./-]+\.[a-zA-Z]{1,5}\b|[\w-]+/[\w./-]+')
_WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]{2,}')

def _grp(a):
    return 'ex' if a in EXPLORE_A else 'ed' if a in EDIT_A else 'xc' if a in EXEC_A else 'ot'

def _parse_result(rs):
    rs = rs or ''
    out = {}
    out['r_pass'] = 1 if re.search(r'\bpass|passed|\bok\b|success', rs, re.I) else (0 if re.search(r'fail|error|\bko\b', rs, re.I) else -1)
    m = re.search(r'(\d+)\s*match', rs); out['r_matches'] = int(m.group(1)) if m else 0
    m = re.search(r'in\s*(\d+)\s*file|(\d+)\s*files?', rs); out['r_files'] = int(next(g for g in (m.groups() if m else []) if g)) if m else 0
    m = re.search(r'\((\d+)\+/(\d+)-\)', rs); out['r_ladd'] = int(m.group(1)) if m else 0; out['r_ldel'] = int(m.group(2)) if m else 0
    m = re.search(r'(\d+)\s*tests?', rs); out['r_tests'] = int(m.group(1)) if m else 0
    m = re.search(r'\((\d+)L\)', rs); out['r_readlines'] = int(m.group(1)) if m else 0
    return out

def build_rich(sample):
    cp = (sample.get("current_prompt") or "")
    hist = sample.get("history", []) or []
    d = {}
    for k, rgx in _KW.items(): d[k] = int(bool(rgx.search(cp)))
    d['p_q'] = int(bool(_Q.search(cp))); d['p_uncert'] = int(bool(_UNCERT.search(cp)))
    d['p_2nd'] = int(bool(_SECOND.search(cp))); d['p_len'] = len(cp); d['p_words'] = len(cp.split())
    p_paths = set(_PATHTOK.findall(cp)); p_words = set(w.lower() for w in _WORD.findall(cp))
    d['p_haspath'] = int(len(p_paths) > 0)
    sm = sample.get("session_meta", {}) or {}; ws = sm.get("workspace", {}) or {}
    openf = ws.get("open_files", []) or []
    open_base = set()
    for of in openf:
        open_base.add(of.lower()); open_base.add(of.split('/')[-1].lower())
    d['ref_openfile'] = int(any(pp.lower() in open_base or pp.split('/')[-1].lower() in open_base for pp in p_paths))
    past_arg_tokens = set(); past_actions = []; past_results = []; user_turns = []
    for h in hist:
        role = h.get('role')
        if role == 'assistant_action':
            past_actions.append(h.get('name'))
            a = h.get('args', {}) or {}
            for v in a.values():
                for w in _WORD.findall(str(v)): past_arg_tokens.add(w.lower())
                for pp in _PATHTOK.findall(str(v)): past_arg_tokens.add(pp.lower()); past_arg_tokens.add(pp.split('/')[-1].lower())
            past_results.append(h.get('result_summary', ''))
        elif role == 'user':
            user_turns.append(h.get('content', ''))
    d['ref_hist_path'] = int(any(pp.lower() in past_arg_tokens or pp.split('/')[-1].lower() in past_arg_tokens for pp in p_paths))
    d['ref_hist_token'] = int(len(p_words & past_arg_tokens) > 0)
    d['n_shared_hist_tok'] = len(p_words & past_arg_tokens)
    pr = _parse_result(past_results[-1] if past_results else '')
    for k, v in pr.items(): d['last_' + k] = v
    d['last_is_edit'] = int(bool(past_actions) and _grp(past_actions[-1]) == 'ed')
    d['last_is_exec'] = int(bool(past_actions) and _grp(past_actions[-1]) == 'xc')
    d['last_is_explore'] = int(bool(past_actions) and _grp(past_actions[-1]) == 'ex')
    last3 = [_grp(a) for a in past_actions[-3:]]
    d['r3_explore'] = last3.count('ex'); d['r3_edit'] = last3.count('ed'); d['r3_exec'] = last3.count('xc')
    def since(pred):
        for i, a in enumerate(reversed(past_actions)):
            if pred(a): return i
        return len(past_actions) + 1
    d['n_since_edit'] = since(lambda a: _grp(a) == 'ed')
    d['n_since_exec'] = since(lambda a: _grp(a) == 'xc')
    d['n_user_turns'] = len(user_turns)
    return d

# ---- 타겟피처 (G2 오답분석 log-odds) ----
_ERR = re.compile(r'([a-z]+error|nonetype|traceback|exception|크래시|crash|죽는|죽어|터졌|터진|안 ?돼|안 ?되|실패|failing|broke|broken|hitting|뜨는데|익셉션|오류|에러)', re.I)
_STUCK = re.compile(r'(모르겠|모르는|감이 ?안|어디서부터|어떻게 ?해야|막막|도와줘|도와주|도와줄|헷갈|판단이|고민|no idea|not sure|help me|어느 ?쪽|긴가민가)', re.I)
_PLAN = re.compile(r'(단계|순서|쪼개|잡아줘|잡아 ?줄|세워|정리해|플랜|계획|로드맵|스텝|그림부터|전체 ?그림|손대기 ?전|before (we|you) ?touch|break (this|it|the)?.{0,8}down|break down|step.?by.?step|lay out|outline|plan (this|it|out))', re.I)
_WEB = re.compile(r'(공식|문서|최신|정석|베스트 ?프랙|프랙티스|관례|컨벤션|표준|보편|일반적으로|요즘|보통|찾아봐|검색|구글|룩업|best ?practice|docs|documentation|standard (way|approach)|recommended|idiomatic|look (it|this|that) up|convention|호환)', re.I)

def build_targeted(sample):
    cp = (sample.get("current_prompt") or "").lower()
    hist = sample.get("history", []) or []
    acts = [h.get("name") for h in hist if h.get("role") == "assistant_action"]
    prev = acts[-1] if acts else "none"
    first = int(len(acts) == 0)
    f_err = int(bool(_ERR.search(cp))); f_stuck = int(bool(_STUCK.search(cp)))
    f_plan = int(bool(_PLAN.search(cp))); f_web = int(bool(_WEB.search(cp)))
    return {
        "f_err": f_err, "f_stuck": f_stuck, "f_ask_any": int(f_err or f_stuck),
        "f_plan": f_plan, "f_web": f_web,
        "f_plan_first": int(f_plan and first),
        "f_web_invest": int(f_web and prev in ("grep_search","read_file","list_directory","glob_pattern")),
    }

def _src_au(rid):
    return int(str(rid).startswith("sess_au"))

def records_to_frame(records):
    """raw 레코드 리스트 -> 원시 피처 DataFrame (tier z-score 제외). 'id' 포함."""
    rows = []
    for s in records:
        d = build_ext(s); d.update(build_rich(s)); d.update(build_targeted(s))
        d["id"] = s.get("id"); d["src_au"] = _src_au(s.get("id"))
        rows.append(d)
    return pd.DataFrame(rows)

# ---- tier 상대 z-score (train 통계로 fit; NEWF의 budget/loc z) ----
def fit_tier_stats(df):
    stats = {}
    for col in ["budget", "loc"]:
        gg = df.groupby("user_tier")[col]
        stats[col] = {"mean": gg.mean().to_dict(),
                      "std": gg.std().replace(0, 1).fillna(1).to_dict(),
                      "gmean": float(df[col].mean()), "gstd": float(df[col].std() or 1.0)}
    return stats

def apply_tier_stats(df, stats):
    out = df.copy()
    for col, name in [("budget", "budget_rel_tier_z"), ("loc", "loc_rel_tier_z")]:
        st = stats[col]
        m = df["user_tier"].map(st["mean"]).fillna(st["gmean"])
        s = df["user_tier"].map(st["std"]).replace(0, 1).fillna(st["gstd"] or 1.0)
        out[name] = ((df[col] - m) / s).fillna(0.0)
    return out

TREE_FEATS = EXT_NUM + CAT_COLS + RICH_COLS + NEWF_COLS + TGT_COLS
LIN_NUM = EXT_NUM + RICH_COLS + NEWF_COLS   # LIN StandardScaler 대상(수치)
CLASSES = ["plan_task", "ask_user", "web_search"]
