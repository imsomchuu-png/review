"""
학점은행제 학습설계표 자동 검증 대시보드 (Streamlit)
표준교육과정 텍스트(curriculums/*.txt)와 붙여넣은 설계표를 대조합니다.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 상수 / 타입
# ---------------------------------------------------------------------------

DB_CAT_REQ = "전필"  # 표준 파일의 전공필수
DB_CAT_ELEC = "전선"  # 전공선택
DB_CAT_LIB = "교양"
DB_CAT_GEN = "일반"  # 자유선택·일반 자격증 등 (전공·교양 외)

DegreeKind = Literal["학사", "전문학사", "타전공학사", "타전공전문학사", "미상"]

SEM_CREDIT_CAP = 24
SEM_COURSE_CAP = 8
YEAR_CREDIT_CAP = 42
YEAR_COURSE_CAP = 14
CLASS_CREDIT_MIN = 18  # 자격증·독학사 제외 수업 이수 최소
INST_CAP_BA = 105
INST_CAP_AA = 60

_EXCEL_PARSE_JUNK_LINE_KEYWORDS: Tuple[str, ...] = (
    "망전공",
    "학위취득",
    "총학점",
    "총 학점",
    "합계",
    "진행여부",
    "학위신청",
    "이수학점",
    "교육부",
    "학습자등록",
    "학점인정신청",
    "보유학점",
    "경기대이수학점",
    "이수학점중학점은행제",
    "학위신청기간",
)


@dataclass
class ParsedCourse:
    name_raw: str
    user_category: Optional[str]
    credits: float
    year: int
    semester: int
    institution: Optional[str] = None
    learning_mode: str = "수업"  # 수업 | 자격증 | 독학사
    is_outsource_other_institution: bool = False  # 타교육기관 외주 — Rule4 본원 합산 제외


@dataclass
class ParsedCertificate:
    name: str
    credits: float
    apply_kind: Optional[str] = None
    구분: str = ""  # 파싱 후 확정: 전필 / 교양 / 일반 (비전공 자격증은 일반)


@dataclass
class ParsedPlan:
    raw_summary: str = ""
    degree_kind: DegreeKind = "미상"
    major_query: str = ""
    courses: List[ParsedCourse] = field(default_factory=list)
    certificates: List[ParsedCertificate] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)
    # 브리핑용
    degree_display: str = ""
    expected_graduation: Optional[str] = None  # 예: "2028-02"
    expected_graduation_explicit: bool = False  # True면 설계표 '학위취득예정일 YYYY-MM'에서 추출
    summary_row: Optional[Dict[str, float]] = None  # 전필/전선/교양/합계 등
    input_text: str = ""  # 붙여넣기 원문 — 표준교육과정 파일명 하드매칭용


@dataclass
class CheckResult:
    rule_id: str
    title: str
    ok: bool
    detail: str = ""
    guides: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 문자열 정규화·매칭
# ---------------------------------------------------------------------------

_ROMAN_MAP = str.maketrans({"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV"})


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_ROMAN_MAP)
    s = re.sub(r"\s+", "", s)
    s = s.replace("(", "").replace(")", "")
    return s.lower()


def normalize_course_key(name: str) -> str:
    n = normalize_text(name)
    for token in ("전공필수", "전공선택", "전필", "전선", "교양"):
        n = n.replace(token.lower(), "")
    return n


def similarity_tokens(a: str, b: str) -> float:
    """간단 일치 점수 (부분 포함 + 정규화 일치)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    # 단어 겹침
    sa = set(re.findall(r"[가-힣a-z0-9]+", a))
    sb = set(re.findall(r"[가-힣a-z0-9]+", b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


_MATCH_SUFFIX_TAIL = re.compile(r"(개론|기초|입문)$")

# 설계표·전적대 등 변형 과목명 → 표준교육과정에 실제 존재하는 과목명(전공 필드).
_COURSE_DB_CANONICAL_ALIASES: Dict[str, str] = {
    normalize_course_key("IT융합공학개론"): "공학과인공지능",
    normalize_course_key("데이터통신개론"): "데이터통신",
}


def _iter_course_match_surfaces(cleaned_name: str) -> List[str]:
    """끝의 개론·기초·입문 접미를 반복 제거한 표면들 (DB 매칭 재시도)."""
    seen: List[str] = []
    cur = unicodedata.normalize("NFKC", cleaned_name.strip())
    if not cur:
        return []
    for _ in range(5):
        if cur not in seen:
            seen.append(cur)
        nxt = _MATCH_SUFFIX_TAIL.sub("", cur).strip()
        if not nxt or nxt == cur:
            break
        cur = nxt
    return seen


def best_course_match(name: str, candidates: Iterable[str]) -> Optional[str]:
    """후보 과목명 중 가장 유사한 표준명 선택."""
    cands = list(candidates)
    if not cands:
        return None
    cand_set = set(cands)
    cleaned = course_name_for_db_match(name).strip()
    canon_map = _COURSE_DB_CANONICAL_ALIASES

    for surf in _iter_course_match_surfaces(cleaned):
        nk_alias = normalize_course_key(surf)
        if nk_alias in canon_map:
            tgt = canon_map[nk_alias]
            if tgt in cand_set:
                return tgt

    best: Optional[str] = None
    score = 0.55
    for surf in _iter_course_match_surfaces(cleaned):
        nk = normalize_course_key(surf)
        if not nk:
            continue
        for c in cands:
            ck = normalize_course_key(c)
            sc = similarity_tokens(nk, ck)
            if sc > score:
                score, best = sc, c
    return best


def course_name_for_db_match(name_raw: str) -> str:
    """DB 매칭 전 과목명 정규화 (타교육기관·재수강 접두 제거)."""
    nm = name_raw.strip()
    nm = re.sub(r"\s*타\s*교육기관\s*", "", nm)
    nm = re.sub(r"^(?:재수강|재이수)\s*", "", nm)
    return nm.strip()


def is_outsource_course_name(name_raw: str) -> bool:
    return bool(re.search(r"타\s*교육기관", name_raw))


_PHANTOM_NAME_SUBSTR = ("학점", "합계", "총", "진행")


def is_phantom_parsed_course_name(name_raw: str) -> bool:
    """요약·상태 텍스트가 과목명으로 오인된 경우 제외 (예: 총학점 → 2학점)."""
    n = unicodedata.normalize("NFKC", name_raw or "").strip()
    if len(n) <= 1:
        return True
    return any(tok in n for tok in _PHANTOM_NAME_SUBSTR)


def map_user_cat_to_db(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    c = re.sub(r"\s+", "", cat.strip())
    mapping = {
        "전필": DB_CAT_REQ,
        "전공필수": DB_CAT_REQ,
        "필수": DB_CAT_REQ,
        "전선": DB_CAT_ELEC,
        "전공선택": DB_CAT_ELEC,
        "선택": DB_CAT_ELEC,
        "교양": DB_CAT_LIB,
        "일반교양": DB_CAT_LIB,
        "핵심교양": DB_CAT_LIB,
        "일반": DB_CAT_GEN,
    }
    return mapping.get(c, None)


# ---------------------------------------------------------------------------
# 표준교육과정 로딩
# ---------------------------------------------------------------------------

_SKIP_LINES = {
    "교수",
    "요목 개설교육",
    "훈련기관",
    "교육목표",
    "세부교육과정표",
}


def _read_major_title(first_lines: str) -> Tuple[str, str]:
    """파일명 stem 과 첫 '전공명' 줄에서 전공 표시명 추출."""
    m = re.search(r"전공명\s*(.+?)(?:\(|$)", first_lines)
    if m:
        title = m.group(1).strip()
        title = re.sub(r"\s+", " ", title)
        return title, title
    return "", ""


def parse_single_curriculum_file(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    표준교육과정 텍스트: '전공필수/전공선택' 줄 다음 줄이 항상 과목명.
    과목명은 NFKC + strip 로 정규화해 DB 문자열 매칭률을 높임.
    """
    raw_text = path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
    lines = raw_text.splitlines()
    title_blob = "\n".join(lines[: min(40, len(lines))])
    major_title, _ = _read_major_title(title_blob)

    stem = path.stem
    degree_guess = "전문학사" if "전문학사" in stem else ("학사" if "학사" in stem else "미상")

    rows: List[Dict[str, str]] = []
    i = 0
    while i < len(lines):
        line_norm = unicodedata.normalize("NFKC", lines[i]).strip()
        if line_norm in ("전공필수", "전공선택"):
            cat = DB_CAT_REQ if line_norm == "전공필수" else DB_CAT_ELEC
            if i + 1 >= len(lines):
                i += 1
                continue
            nxt_raw = unicodedata.normalize("NFKC", lines[i + 1]).strip()
            course_name = nxt_raw.strip()
            if (
                course_name
                and course_name not in ("전공필수", "전공선택")
                and course_name not in _SKIP_LINES
                and not course_name.startswith("학점")
            ):
                rows.append(
                    {
                        "file_stem": stem,
                        "degree_guess": degree_guess,
                        "major_title": major_title.strip() if major_title else "",
                        "course_name": course_name,
                        "category": cat,
                    }
                )
        i += 1

    meta = {"stem": stem, "degree_guess": degree_guess, "major_title": major_title}
    df = pd.DataFrame(rows)
    return df, meta


def load_curriculum_database(curriculum_dir: Path) -> Tuple[pd.DataFrame, List[Dict]]:
    """curriculums 폴더의 모든 .txt 를 통합 DataFrame 으로 적재."""
    files = sorted(curriculum_dir.glob("*.txt"))
    frames: List[pd.DataFrame] = []
    metas: List[Dict] = []
    for f in files:
        try:
            df, meta = parse_single_curriculum_file(f)
            if not df.empty:
                frames.append(df)
            metas.append({"path": str(f), **meta})
        except OSError:
            continue
    if not frames:
        return pd.DataFrame(), metas
    return pd.concat(frames, ignore_index=True), metas


_MAJOR_KEYWORD_CANON: Tuple[Tuple[str, str], ...] = (
    # 긴·구체 키워드 먼저 (희망전공컴퓨터공학 붙여쓰기·교차 오판 방지)
    ("컴퓨터공학", "컴퓨터공학"),
    ("컴공", "컴퓨터공학"),
    ("사회복지학", "사회복지학"),
    ("사회복지", "사회복지학"),
    ("체육학", "체육학"),
    ("체육", "체육학"),
    ("경영학", "경영학"),
    ("경영", "경영학"),
    ("인공지능학", "인공지능학"),
    ("인공지능", "인공지능학"),
    ("관광경영학", "관광경영"),
    ("관광경영", "관광경영"),
    ("정보통신", "정보통신"),
    ("정보처리", "정보처리"),
    ("레저스포츠", "레저스포츠"),
)


def detect_major_keyword_from_degree_text(degree_text: str) -> str:
    """희망학위 텍스트에서 핵심 전공 키워드 탐지."""
    t = unicodedata.normalize("NFKC", degree_text or "")
    t_ns = re.sub(r"\s+", "", t)
    for kw, canon in _MAJOR_KEYWORD_CANON:
        if kw in t or kw in t_ns:
            return canon
    return ""


def resolve_major_canon_from_plan(plan: ParsedPlan) -> str:
    """전공 확정: 희망전공/학위 표기 우선, 입력 전체 스캔은 최후 fallback."""
    for src in (str(plan.major_query or ""), str(plan.degree_display or "")):
        k = detect_major_keyword_from_degree_text(src)
        if k:
            return k
    return detect_major_keyword_from_degree_text(str(plan.input_text or ""))


def resolve_curriculum_filename(degree_text: str) -> str:
    """목표 학위·전공 텍스트로 표준교육과정 파일명을 고정 매핑한다."""
    t = unicodedata.normalize("NFKC", degree_text or "")
    clean_degree = re.sub(r"\s+", "", t)

    db_filename = "경영학표준교육과정.txt"

    # 컴퓨터공학·컴공은 사회복지 등 다른 전공 키워드보다 우선
    if "컴퓨터공학" in clean_degree or "컴공" in clean_degree:
        return "컴퓨터공학표준교육과정.txt"

    if "체육학" in clean_degree:
        db_filename = "체육학표준교육과정.txt"
    elif "체육" in clean_degree:
        db_filename = "체육학표준교육과정.txt"
    elif "사회복지학" in clean_degree:
        db_filename = "사회복지학표준교육과정.txt"
    elif "사회복지전문" in clean_degree:
        db_filename = "사회복지전문학사표준교육과정.txt"
    elif "경영전문" in clean_degree:
        db_filename = "경영전문학사표준교육과정.txt"
    elif "관광경영전문" in clean_degree:
        db_filename = "관광경영전문학사.txt"
    elif "관광경영학" in clean_degree:
        db_filename = "관광경영학표준교육과정.txt"
    elif "레저스포츠" in clean_degree:
        db_filename = "레저스포츠전문학사표준교육과정.txt"
    elif "인공지능" in clean_degree:
        db_filename = "인공지능학표준교육과정.txt"
    elif "정보처리" in clean_degree:
        db_filename = "정보처리전문학사표준교육과정.txt"
    elif "정보통신" in clean_degree:
        db_filename = "정보통신전문학사표준교육과정.txt"

    return db_filename


def pick_curriculum_subset(
    df: pd.DataFrame, metas: List[Dict], plan: ParsedPlan, kind: DegreeKind
) -> pd.DataFrame:
    """표준교육과정: 파일명을 하드 매핑으로만 선택한다."""
    del kind  # 학위 종류와 무관하게 위 매핑만 사용
    if df.empty:
        return df

    # IMPORTANT:
    # 표준과정 선택은 희망전공/학위 표기만 사용한다.
    # 과목 본문(input_text)에는 타전공 과목명이 섞여 있어 전공 오판(예: 인공지능학 -> 사회복지학)이 발생할 수 있다.
    degree_blob = " ".join(
        [
            str(plan.major_query or ""),
            str(plan.degree_display or ""),
            str(plan.raw_summary or ""),
        ]
    )
    db_filename = resolve_curriculum_filename(degree_blob)
    stem = Path(db_filename).stem

    sub = df[df["file_stem"] == stem].copy()
    if not sub.empty:
        return sub

    for m in metas:
        if Path(str(m.get("path", ""))).name == db_filename:
            return df[df["file_stem"] == str(m.get("stem", ""))].copy()

    return sub


# ---------------------------------------------------------------------------
# 설계표 텍스트 파싱 (붙여넣기·붙어쓰기·날짜 혼입 방어)
# ---------------------------------------------------------------------------

_DEGREE_PATTERNS = [
    (re.compile(r"전문학사\s*타전공|전문학사타전공", re.I), "타전공전문학사"),
    (re.compile(r"타전공\s*전문학사|타전공전문학사", re.I), "타전공전문학사"),
    (re.compile(r"학사\s*타전공|학사타전공", re.I), "타전공학사"),
    (re.compile(r"타전공\s*학사|타전공학사", re.I), "타전공학사"),
    (re.compile(r"전문학사", re.I), "전문학사"),
    (re.compile(r"학사", re.I), "학사"),
]

_MAJOR_HINT = re.compile(
    r"(?:희망\s*전공|신청\s*전공|전공|학위\s*과정)\s*[:：]?\s*(.+?)(?:\n|$)",
    re.I,
)

_SEM_HEAD = re.compile(
    r"(?P<y>\d{4})\s*년?\s*(?P<s>[12])\s*학기"
    r"|(?P<y2>\d{4})\s*[-_/]\s*(?P<s2>[12])(?:\s*학기)?"
    r"|(?P<y3>\d{4})\s*학년도\s*(?P<s3>[12])\s*학기",
    re.I,
)

_CERT_LINE = re.compile(
    r"^\s*(?P<name>.+?)\s+"
    r"(?P<cred>\d+(?:\.\d+)?)\s*학점?"
    r"(?:.*?(?:적용|구분|비고)[:：]\s*(?P<app>.+?))?\s*$",
    re.I,
)

# 하단 요약 패턴 (과목으로 오인 금지)
_SUMMARY_COMPACT = re.compile(
    r"전필\s*(\d+)\s*전선\s*(\d+)\s*교양\s*(\d+)(?:\s*일반\s*(\d+))?\s*합계\s*(\d+)",
    re.I,
)
_SUMMARY_COMPACT_NS = re.compile(
    r"전필(\d+)전선(\d+)교양(\d+)(?:일반(\d+))?합계(\d+)",
    re.I,
)

_DATE_STRIP = re.compile(
    r"\d{4}\s*년\s*[12]\s*학기"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{4}[./-]\d{1,2}[./-]\d{1,2}"
    r"|\d{4}\s*년\s*\d{1,2}\s*월",
    re.I,
)

_STATUS_STRIP = re.compile(
    r"^(?:수강예정|진행예정|이수완료|이수예정|재수강|대기중|등록예정|폐강|학점인정\s*필요|학점인정필요|학점인정)\s*",
    re.I,
)

_GRADUATION_HINT = re.compile(
    r"(?:취득\s*예정|졸업\s*예정|졸업예정|수료\s*예정|예상\s*취득)\s*[:：]?\s*(\d{4})\s*[-년.\s/]\s*(\d{1,2})",
    re.I,
)
# 설계표 첫머리 등에 명시된 학위취득예정일(YYYY-MM)
_DESIGN_SHEET_GRAD_DATE = re.compile(r"학위취득예정일\s*[：:]?\s*(\d{4}-\d{2})")
# 접두 없이 '명칭+끝자리 학점'만 있는 자격증 후보 (소방안전관리자20 등)
_CERTISH_NAME_HINT = re.compile(
    r"(관리자|안전|산업기사|기능사|기능장|정보처리|컴퓨터활용|컴활|기사|자격|민간자격|국가자격|소방)",
)


def _strip_dates_and_sem_tokens(s: str) -> str:
    """줄 안의 날짜·학기 표기 제거 (숫자를 학점으로 오인 방지)."""
    t = _DATE_STRIP.sub("", s)
    return t


def _strip_leading_status_loop(s: str) -> str:
    """상태 접두 반복 제거."""
    cur = s
    for _ in range(8):
        nxt = _STATUS_STRIP.sub("", cur).strip()
        if nxt == cur:
            break
        cur = nxt
    return cur


# 과목명 앞 '찌꺼기' 고정 패턴 (날짜·상태가 과목명에 붙어 DB 매칭 실패 나는 경우)
_JUNK_PREFIX_STATUS_DATE = re.compile(r"수강예정\s*\d{4}-\d{2}-\d{2}", re.I)
_JUNK_PREFIX_SEM_PROGRESS = re.compile(
    r"진행예정\s*\d{4}\s*년\s*[12]\s*학기\s*진행예정|진행예정\d{4}년\s*[12]\s*학기\s*진행예정",
    re.I,
)


def strip_course_boilerplate_prefixes(s: str) -> str:
    """
    엑셀/포털에서 붙은 상태+날짜+학기 문자열을 과목 앞에서 제거.
    (남은 문자열에서 끝 학점 분리 → 순수 과목명)
    """
    t = unicodedata.normalize("NFKC", s.strip())
    for _ in range(12):
        old = t
        t = _JUNK_PREFIX_STATUS_DATE.sub("", t)
        t = _JUNK_PREFIX_SEM_PROGRESS.sub("", t)
        t = t.strip()
        if t == old:
            break
    return t


# 과목명 끝(학점 직전)으로 올 수 있는 문자 — 로마숫자(ⅠⅡ…)·글·영문·글 내 숫자(1급) 등
_COURSE_TAIL_BEFORE_CREDIT = r"0-9가-힣a-zA-Z급ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ"


def split_trailing_credit_digits(compact_tail: str, *, mode: str) -> Tuple[str, Optional[float]]:
    """
    맨 끝 **1자리 또는 2자리** 숫자만 학점으로 분리하고, 그 앞 전체를 이름으로 둠.
    (예: 경영학개론3 → 경영학개론 + 3, 전산회계1급4 → 전산회계1급 + 4)
    """
    s = compact_tail.strip()
    if len(s) < 2:
        return s, None
    if re.search(r"\d{3,}$", s):
        return s, None

    _b = rf"(.+[{_COURSE_TAIL_BEFORE_CREDIT}])"
    mh = re.match(rf"^{_b}(\d)\.5$", s)
    if mh:
        b = mh.group(1).rstrip()
        v = float(mh.group(2)) + 0.5
        mx = 30.0 if mode == "cert" else 9.5
        if v <= mx:
            return b, v

    # 끝 2자리 — 자격 인정 등(예: 전산회계1급14)
    m2 = re.match(rf"^{_b}(\d{{2}})$", s)
    if m2:
        b = m2.group(1).rstrip()
        v = float(m2.group(2))
        if mode == "cert" and 1 <= v <= 30:
            return b, v
        if mode == "course" and 10 <= v <= 16:
            return b, v

    # 끝 1자리 — 일반 과목 (경영학개론3)
    m1 = re.match(rf"^{_b}(\d)$", s)
    if not m1:
        return s, None
    b = m1.group(1).rstrip()
    v = float(m1.group(2))
    if mode == "cert" and 1 <= v <= 9:
        return b, v
    if mode == "course" and 1 <= v <= 6:
        return b, v
    return s, None


def _extract_inline_semester(segment: str) -> Tuple[str, Optional[int], Optional[int]]:
    """문자열 안의 '2026년 2학기' 등을 추출하고 해당 부분은 제거한 나머지 반환."""
    m = _SEM_HEAD.search(segment)
    if not m:
        return segment, None, None
    y = int(m.group("y") or m.group("y2") or m.group("y3"))
    s = int(m.group("s") or m.group("s2") or m.group("s3"))
    rest = segment[: m.start()] + segment[m.end() :]
    rest = _strip_leading_status_loop(rest)
    return rest.strip(), y, s


def _split_munged_records(line: str) -> List[str]:
    """붙여넣기 한 줄에 과목이 여러 개 붙은 경우 분할."""
    line = line.strip()
    if not line:
        return []
    # 요약 한 줄은 분할하지 않음
    if _SUMMARY_COMPACT_NS.search(re.sub(r"\s+", "", line)) or _SUMMARY_COMPACT.search(line):
        return [line]
    # 상태 키워드 앞에서 분리 (다음 레코드 시작)
    parts = re.split(r"(?=(?:(?:수강예정|진행예정|이수완료|이수예정)\s*\d{4}))", line)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [line]


def parse_compact_summary_line(line: str) -> Optional[Dict[str, float]]:
    raw = re.sub(r"\s+", "", line.strip())
    m = _SUMMARY_COMPACT_NS.search(raw)
    if not m:
        m = _SUMMARY_COMPACT.search(line.strip())
    if not m:
        return None
    gr = m.groups()
    try:
        out: Dict[str, float] = {
            "전필": float(gr[0]),
            "전선": float(gr[1]),
            "교양": float(gr[2]),
            "합계": float(gr[-1]),
        }
        if len(gr) >= 5 and gr[3] not in (None, ""):
            out["일반"] = float(gr[3])
        return out
    except (ValueError, IndexError, TypeError):
        return None


_GARBAGE_LINE_HINTS_RE = (
    re.compile(r"총\s*학점"),
    re.compile(r"진행\s*여부|진행여부"),
    re.compile(r"학위\s*신청\s*기간|학위신청기간"),
    re.compile(r"자격증\s*진행\s*여부|자격증진행여부"),
)


def should_skip_parse_line(raw_line: str) -> bool:
    """
    안내·헤더·요약표가 아닌 쓰레기 줄 제외 (숫자를 학점으로 오인하지 않도록).
    압축 요약 줄(전필25전선24…합계)은 파싱 유지.
    """
    s = raw_line.strip()
    if not s:
        return True
    nospace = re.sub(r"\s+", "", s)
    if _SUMMARY_COMPACT_NS.search(nospace):
        return False
    if _SUMMARY_COMPACT.search(s):
        return False
    for ch in "★※▶◀◇●☆□■◇":
        if ch in s:
            return True
    for pat in _GARBAGE_LINE_HINTS_RE:
        if pat.search(s):
            return True
    if "합계" in s:
        low = s.lower()
        if "전필" not in nospace and "교양" not in nospace:
            if len(s) > 12 or any(x in s for x in ("안내", "유의", "※", "[", "바랍니다", "포함")):
                return True
    noise_tokens = ("과목코드", "이수구분", "교육비", "평점", "담당", "교수", "안내문", "결재란")
    if any((tok.lower() in nospace.lower() or tok in s) for tok in noise_tokens):
        return True
    return False


def remove_all_inline_ws(s: str) -> str:
    """NFKC 후 모든 유니코드 공백·줄바꿈 등 연속 제거(탭·NBSP·일반 공백 포함)."""
    t = unicodedata.normalize("NFKC", s)
    # 제로폭 문자는 \s 에 안 잡히는 경우가 있어 별도 제거
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
    return re.sub(r"\s+", "", t)


# 공백 제거 후 과목/자격 접두 (사용자 지정 패턴)
_COURSE_PREFIX_NOSPACE = re.compile(
    r"^(?:수강예정\d{4}-\d{2}-\d{2}|진행예정\d{4}년\d학기진행예정)"
)
_CERT_PREFIX_NOSPACE = re.compile(r"^(?:학점인정필요|취득예정|학점인정완료)")
_SEM_NOSPACE_HEAD = re.compile(r"^(?:(\d{4})년([12])학기|(\d{4})[-_/]([12])(?:학기)?)")
# 줄/토막 내부 학기 표기 (공백 제거 후)
_SEM_INLINE_NOSPACE = re.compile(r"(?:(\d{4})년([12])학기|(\d{4})[-_/]([12])(?:학기)?)")

_FULL_COURSE_NOSPACE_RE = re.compile(
    r"^(?:수강예정\d{4}-\d{2}-\d{2}|진행예정\d{4}년\d학기진행예정)"
    r"([가-힣A-Za-z0-9Ⅰ-Ⅹ]+?)(\d{1,2})$"
)
_FULL_CERT_NOSPACE_RE = re.compile(
    r"^(?:학점인정필요|취득예정|학점인정완료)([가-힣A-Za-z0-9]+?)(\d{1,2})$"
)


def peel_name_credit_suffix(rest: str, *, max_credit: int) -> Optional[Tuple[str, float]]:
    """문자열 끝에서 2·1자리 학점을 뗀 뒤 앞쪽을 과목/자격명으로 사용."""
    if not rest or len(rest) < 2:
        return None
    for w in (2, 1):
        if len(rest) <= w:
            continue
        tail = rest[-w:]
        if not tail.isdigit():
            continue
        v = int(tail)
        if 1 <= v <= max_credit:
            name = rest[:-w]
            if len(name) < 1:
                continue
            if not re.search(r"[가-힣A-Za-z]", name):
                continue
            if name.isdigit():
                continue
            return name, float(v)
    return None


def extract_cert_from_nospace(nos: str) -> Optional[ParsedCertificate]:
    """(학점인정필요|취득예정|학점인정완료) + 명칭 + 끝 학점 — 오른쪽 학점 우선 분리."""
    if not nos:
        return None
    m = _CERT_PREFIX_NOSPACE.match(nos)
    if not m:
        gm = _FULL_CERT_NOSPACE_RE.match(nos)
        if gm:
            nm, cs = gm.group(1).strip(), int(gm.group(2))
            if 1 <= cs <= 30 and nm:
                return ParsedCertificate(name=nm, credits=float(cs), apply_kind=None)
        return None
    rest = nos[m.end() :]
    p = peel_name_credit_suffix(rest, max_credit=30)
    if p:
        return ParsedCertificate(name=p[0], credits=p[1], apply_kind=None)
    return None


def strip_optional_category_prefix(nos: str) -> Tuple[str, Optional[str]]:
    """전필/전선/교양/일반 접두 제거 후 (나머지, 분류키)."""
    order = (
        ("전공필수", "전필"),
        ("전공선택", "전선"),
        ("전필", "전필"),
        ("전선", "전선"),
        ("교양", "교양"),
        ("일반", "일반"),
    )
    for raw_tok, db_key in order:
        if nos.startswith(raw_tok):
            suf = nos[len(raw_tok) :]
            return suf, map_user_cat_to_db(db_key)
    return nos, None


_INVINCIBLE_COURSE_FULL_RE = re.compile(
    r"^(?:\[(본원|타기관)\])?([가-힣A-Za-z0-9Ⅰ-Ⅹ]+?)(\d{1,2})$"
)
_INVINCIBLE_COURSE_TAIL_RE = re.compile(
    r"(?:\[(본원|타기관)\])?([가-힣A-Za-z0-9Ⅰ-Ⅹ]+?)(\d{1,2})$"
)


def _replace_paren_institution_hakjeom_tail(line: str) -> str:
    """(경기대|본원) 학점인정 완료/필요 꼬리 → [본원], (타기관|전적대) 동일 → [타기관]."""
    line = re.sub(
        r"\((경기대|본원)\)\s*(?:학점인정\s*완료|학점인정\s*필요)?",
        "[본원]",
        line,
    )
    line = re.sub(
        r"\((타기관|전적대)\)\s*(?:학점인정\s*완료|학점인정\s*필요)?",
        "[타기관]",
        line,
    )
    return line


def _line_looks_like_institution_course_after_strip(seg: str) -> bool:
    """(경기대) 학점인정 필요 스포츠심리학3 → 치환 후 [본원]스포츠심리학3 형 수업 줄이면 True (자격증 파서 오인 방지)."""
    t = unicodedata.normalize("NFKC", seg.strip())
    t = _replace_paren_institution_hakjeom_tail(t)
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
    t = re.sub(r"\s+", "", t)
    m = re.match(r"^(?:\[(?:본원|타기관)\])?([가-힣A-Za-z0-9Ⅰ-Ⅹ]+?)(\d{1,2})$", t)
    if not m:
        return False
    cr = int(m.group(2))
    return 1 <= cr <= 9 and len(m.group(1)) >= 2


def preprocess_excel_paste_line_for_course(line: str) -> str:
    """엑셀 복붙 줄에서 날짜·상태·공백·탭을 제거하고 과목+학점 토큰만 남긴다 (요구 순서 준수)."""
    line = unicodedata.normalize("NFKC", line)
    line = line.replace("(학점절삭)", "")
    line = _replace_paren_institution_hakjeom_tail(line)
    line = re.sub(r"20\d{2}년\s*\d학기", "", line)
    line = re.sub(r"20\d{2}-\d{2}-\d{2}", "", line)
    line = re.sub(r"수강예정|진행예정", "", line)
    line = re.sub(r"[\u200b-\u200d\ufeff]", "", line)
    line = re.sub(r"\s+", "", line)
    return line


def extract_course_invincible_chunks(
    clean: str,
) -> List[Tuple[str, float, Optional[str], Optional[str]]]:
    r"""전처리 문자열에서 [본원|타기관]?과목명+학점 반복 분리. 네 번째 값: 기관구분 본원|타기관|None."""
    chunks: List[Tuple[str, float, Optional[str], Optional[str]]] = []
    cur = clean
    while cur:
        mf = _INVINCIBLE_COURSE_FULL_RE.match(cur)
        if mf:
            inst_g, name_piece, ds = mf.group(1), mf.group(2).strip(), mf.group(3)
            ci = int(ds)
            name_body, ucat = strip_optional_category_prefix(name_piece)
            name_body = name_body.strip()
            inst_scope = inst_g if inst_g else None
            if (
                len(name_body) > 1
                and 1 <= ci <= 30
                and re.search(r"[가-힣A-Za-z]", name_body)
                and not name_body.isdigit()
            ):
                chunks.append((name_body, float(ci), ucat, inst_scope))
            break
        mt = _INVINCIBLE_COURSE_TAIL_RE.search(cur)
        if not mt:
            break
        inst_g = mt.group(1)
        name_piece, ds = mt.group(2).strip(), mt.group(3)
        ci = int(ds)
        name_body, ucat = strip_optional_category_prefix(name_piece)
        name_body = name_body.strip()
        inst_scope = inst_g if inst_g else None
        if len(name_body) <= 1 or not (1 <= ci <= 30) or not re.search(r"[가-힣A-Za-z]", name_body) or name_body.isdigit():
            break
        chunks.append((name_body, float(ci), ucat, inst_scope))
        cur = cur[: mt.start()]
    chunks.reverse()
    return chunks


def extract_course_tuple_from_nospace(nos: str) -> Optional[Tuple[str, float, Optional[str]]]:
    """과목명, 학점, 사용자구분.(?:접두) + peel 또는 전체 정규식."""
    if not nos or _CERT_PREFIX_NOSPACE.match(nos):
        return None
    work, ucat = strip_optional_category_prefix(nos)
    m = _COURSE_PREFIX_NOSPACE.match(work)
    if m:
        rest = work[m.end() :]
        p = peel_name_credit_suffix(rest, max_credit=24)
        if p:
            return p[0], p[1], ucat
        return None
    fm = _FULL_COURSE_NOSPACE_RE.match(work)
    if fm:
        return fm.group(1), float(fm.group(2)), ucat
    p2 = peel_name_credit_suffix(work, max_credit=24)
    if p2:
        return p2[0], p2[1], ucat
    return None


def _split_munged_records_nospace(nos: str) -> List[str]:
    """공백·탭 제거 후 한 줄에 붙은 과목을 분리."""
    if not nos:
        return []
    if _SUMMARY_COMPACT_NS.search(nos):
        return [nos]
    parts = re.split(r"(?=(?:(?:수강예정|진행예정|이수완료|이수예정)\d{4}))", nos)
    parts = [p for p in parts if p]
    return parts if parts else [nos]


def _extract_inline_semester_from_nospace(piece_nos: str) -> Tuple[str, Optional[int], Optional[int]]:
    m = _SEM_INLINE_NOSPACE.search(piece_nos)
    if not m:
        return piece_nos, None, None
    y = int(m.group(1) or m.group(3))
    s = int(m.group(2) or m.group(4))
    rest = piece_nos[: m.start()] + piece_nos[m.end() :]
    return rest, y, s


def _strip_leading_course_status_nospace(nos: str) -> str:
    """이수완료 등 접두만 제거(『이수완료2026…』처럼 연도로 이어지는 경우는 유지)."""
    cur = nos
    for _ in range(8):
        if _COURSE_PREFIX_NOSPACE.match(cur) or (cur and _FULL_COURSE_NOSPACE_RE.match(cur)):
            break
        m = re.match(r"^(?:이수완료|이수예정|재수강|대기중|등록예정|폐강)", cur)
        if not m:
            break
        nxt = cur[m.end() :]
        if len(nxt) >= 4 and nxt[:4].isdigit():
            break
        cur = nxt
    return cur


def _institution_from_raw_line(raw_line: str) -> Optional[str]:
    inst_m = re.search(r"\[([^\]]+)\]|\(([^)]+(?:원|교육원|대학|기관))\)", raw_line)
    if inst_m:
        return (inst_m.group(1) or inst_m.group(2) or "").strip()
    return None


def parse_graduation_hint(text: str) -> Tuple[Optional[str], bool]:
    """(졸업·취득 예정 월 YYYY-MM, 설계표 학위취득예정일 직접 추출 여부)."""
    blob = text if isinstance(text, str) else ""
    m_ds = _DESIGN_SHEET_GRAD_DATE.search(blob)
    if m_ds:
        return m_ds.group(1), True
    m = _GRADUATION_HINT.search(blob)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}", False
    m2 = re.search(r"(\d{4})\s*[-년]\s*(\d{1,2})\s*(?:월)?\s*(?:취득|졸업|수료)", blob)
    if m2:
        return f"{m2.group(1)}-{int(m2.group(2)):02d}", False
    return None, False


def parse_certificate_standalone_line(seg: str) -> Optional[ParsedCertificate]:
    """학점인정 접두·자격증 블록 없이 '명칭+끝학점'만 있는 줄 (예: 소방안전관리자 20, 소방안전관리자20)."""
    raw = seg.strip()
    if not raw or should_skip_parse_line(raw):
        return None
    if parse_compact_summary_line(raw):
        return None
    compact = re.sub(r"\s+", "", raw)
    if _SUMMARY_COMPACT_NS.fullmatch(compact) or (
        "전필" in compact and "합계" in compact and "교양" in compact and len(compact) < 80
    ):
        return None
    nos = remove_all_inline_ws(raw)
    if extract_cert_from_nospace(nos):
        return None
    if parse_certificate_line_loose(raw):
        return None
    if parse_munged_certificate_segment(raw):
        return None
    m = re.match(r"^(.+?)(\d{1,2})$", nos)
    if not m:
        return None
    name = m.group(1).strip()
    cr = int(m.group(2))
    if not (1 <= cr <= 30) or len(name) < 3:
        return None
    if not re.search(r"[가-힣A-Za-z]", name):
        return None
    # 고학점은 자격증 인정 단위로 가정, 저학점은 자격증형 명칭일 때만
    if cr >= 10:
        pass
    elif _CERTISH_NAME_HINT.search(name):
        pass
    else:
        return None
    return ParsedCertificate(name=name, credits=float(cr), apply_kind=None)


def parse_munged_course_segment(segment: str) -> Optional[Dict[str, object]]:
    """붙어 있는 과목 한 건: 수강예정2026-05-14경영학개론3 등."""
    raw = segment.strip()
    if not raw or len(raw) < 3:
        return None
    raw = unicodedata.normalize("NFKC", raw)
    raw = _replace_paren_institution_hakjeom_tail(raw)
    if "합계" in raw and "전필" in raw and "교양" in raw:
        return None

    inst_m = re.search(r"\[([^\]]+)\]|\(([^)]+(?:원|교육원|대학|기관))\)", raw)
    inst = (inst_m.group(1) or inst_m.group(2)).strip() if inst_m else None

    work = raw
    work = re.sub(r"\[([^\]]+)\]", "", work)
    work = re.sub(r"\([^)]*(?:원|교육원|대학)[^)]*\)", "", work)

    learning_mode = "수업"
    if "독학사" in work:
        learning_mode = "독학사"

    work = strip_course_boilerplate_prefixes(work)
    work = _strip_dates_and_sem_tokens(work)
    work = _strip_leading_status_loop(work)

    user_cat: Optional[str] = None
    cm = re.match(r"^(전필|전선|교양|전공필수|전공선택|일반)", work)
    if cm:
        user_cat = map_user_cat_to_db(cm.group(1))
        work = work[cm.end() :].strip()

    name_body, credit = split_trailing_credit_digits(work, mode="course")
    if credit is None or not name_body:
        return None
    if not re.search(r"[가-힣a-zA-Z]", name_body):
        return None

    return {
        "name_raw": name_body.strip(),
        "user_category": user_cat,
        "credits": float(credit),
        "institution": inst,
        "learning_mode": learning_mode,
    }


def parse_glued_cert_line(seg: str) -> Optional[ParsedCertificate]:
    """붙여쓰기 자격 줄 — 공백·탭 제거 후 학점인정필요/취득예정 + 끝자리 학점."""
    return extract_cert_from_nospace(remove_all_inline_ws(seg))


def parse_munged_certificate_segment(segment: str) -> Optional[ParsedCertificate]:
    """학점인정 필요전산회계1급4 형태 (접두+끝학점 패턴 또는 끝자리 학점 분리)."""
    w = segment.strip()
    glued = parse_glued_cert_line(w)
    if glued:
        return glued

    w = _strip_dates_and_sem_tokens(w)
    w = _strip_leading_status_loop(w)
    w = re.sub(r"^(?:국가\s*자격|자격증\s*)", "", w).strip()

    name_body, credit = split_trailing_credit_digits(w, mode="cert")
    if credit is None or not name_body or len(name_body) < 2:
        return None
    app = None
    ap = re.search(r"(?:적용|구분|비고)\s*[:：]\s*(.+)$", name_body)
    if ap:
        app = ap.group(1).strip()
        name_body = name_body[: ap.start()].strip()

    return ParsedCertificate(name=name_body.strip(), credits=float(credit), apply_kind=app)


def parse_course_line(line: str) -> Optional[Dict[str, object]]:
    """공백·표 형식 한 줄 (레거시). 숫자 열은 학점 후보만 허용."""
    raw = line.strip()
    if not raw or raw.startswith("#") or len(raw) < 2:
        return None
    if should_skip_parse_line(raw):
        return None
    if re.match(r"^(합계|소계|총\s*계|과목\s*명|이수구분)", raw):
        return None

    compact = re.sub(r"\s+", "", raw)
    if _SUMMARY_COMPACT_NS.fullmatch(compact) or (
        "전필" in compact and "합계" in compact and "교양" in compact and len(compact) < 80
    ):
        return None

    # 붙어쓰기 패턴 우선
    munged_try = parse_munged_course_segment(raw)
    if munged_try:
        return munged_try

    inst = None
    inst_m = re.search(r"\[([^\]]+)\]|\(([^)]+(?:원|교육원|대학|기관))\)", raw)
    if inst_m:
        inst = (inst_m.group(1) or inst_m.group(2) or "").strip()

    learning_mode = "수업"
    if "독학사" in raw:
        learning_mode = "독학사"

    work = raw
    work = re.sub(r"\[([^\]]+)\]", "", work)
    work = re.sub(r"\([^)]*(?:원|교육원|대학)[^)]*\)", "", work)
    work = strip_course_boilerplate_prefixes(work)

    patterns: List[Tuple[re.Pattern, str]] = [
        (
            re.compile(
                r"^\s*\d+\s*[.)]?\s*(.+?)\s+(전필|전선|전공필수|전공선택|교양|일반교양|핵심교양|일반)\s+(\d+(?:\.\d+)?)\s*$"
            ),
            "indexed",
        ),
        (
            re.compile(
                r"^\s*(.+?)\s+(전필|전선|전공필수|전공선택|교양|일반교양|핵심교양|일반)\s+(\d+(?:\.\d+)?)\s*$"
            ),
            "named_first",
        ),
        (
            re.compile(r"^\s*(.+?)\s+(\d+(?:\.\d+)?)\s*학점\s*$"),
            "credit_suffix",
        ),
        (
            re.compile(r"^\s*(\d+(?:\.\d+)?)\s*학점\s+(.+?)\s*$"),
            "credit_first",
        ),
        (
            re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(\d+(?:\.\d+)?)\s*$"),
            "pipe3",
        ),
    ]

    for pat, kind in patterns:
        m = pat.match(work.strip())
        if not m:
            continue
        name: str
        cat: Optional[str]
        cred_s: str
        if kind == "indexed":
            name, cat, cred_s = m.group(1).strip(), m.group(2).strip(), m.group(3)
        elif kind == "named_first":
            name, cat, cred_s = m.group(1).strip(), m.group(2).strip(), m.group(3)
        elif kind == "credit_suffix":
            name, cat, cred_s = m.group(1).strip(), None, m.group(2)
            inner = re.match(r"(.+?)\s+(전필|전선|교양|전공필수|전공선택|일반)$", name)
            if inner:
                name, cat = inner.group(1).strip(), inner.group(2).strip()
        elif kind == "credit_first":
            cred_s, name = m.group(1), m.group(2).strip()
            cat = None
            inner = re.match(r"(.+?)\s+(전필|전선|교양|전공필수|전공선택|일반)$", name)
            if inner:
                name, cat = inner.group(1).strip(), inner.group(2).strip()
        elif kind == "pipe3":
            name, cat, cred_s = m.group(1).strip(), m.group(2).strip(), m.group(3)
        else:
            continue

        try:
            credits = float(cred_s)
        except ValueError:
            continue

        # 학점 후보 검증: 날짜 조각·행 번호 제외
        if credits >= 100:
            continue
        if credits > 24 and learning_mode == "수업":
            continue
        if not _looks_like_course_credit(credits, cred_s):
            continue

        return {
            "name_raw": name.strip(),
            "user_category": cat,
            "credits": credits,
            "institution": inst,
            "learning_mode": learning_mode,
        }

    cols = re.split(r"\t+|\s{2,}", work)
    cols = [c.strip() for c in cols if c.strip()]
    if len(cols) >= 3:
        nums_idx = [i for i, c in enumerate(cols) if re.fullmatch(r"\d+(?:\.\d+)?", c)]
        if nums_idx:
            cred_col = nums_idx[-1]
            cred_s = cols[cred_col]
            left = cols[:cred_col]
            if len(left) >= 2:
                name_col, cat_col = left[0], left[1]
            elif len(left) == 1:
                name_col, cat_col = left[0], None
            else:
                return None
            try:
                credits = float(cred_s)
            except ValueError:
                return None
            if credits >= 100 or credits > 24:
                return None
            if not _looks_like_course_credit(credits, cred_s):
                return None
            return {
                "name_raw": name_col,
                "user_category": cat_col,
                "credits": credits,
                "institution": inst,
                "learning_mode": learning_mode,
            }

    return None


def _looks_like_course_credit(credits: float, raw_s: str) -> bool:
    """표에서 연도·순번 열이 학점으로 잡히는 것 완화."""
    if credits != int(credits) and credits not in (1.5, 2.5, 3.5):
        return False
    iv = int(credits)
    if iv >= 1900 and iv <= 2100:
        return False
    if iv >= 100:
        return False
    return 0.5 <= credits <= 24


def _default_year_sem(text: str) -> Tuple[int, int]:
    m = _SEM_HEAD.search(text)
    if m:
        y = m.group("y") or m.group("y2") or m.group("y3")
        s = m.group("s") or m.group("s2") or m.group("s3")
        return int(y), int(s)
    return 2026, 1


def split_semester_blocks(text: str) -> List[Tuple[Tuple[int, int], str]]:
    matches = list(_SEM_HEAD.finditer(text))
    if not matches:
        y, s = _default_year_sem(text)
        return [((y, s), text)]

    blocks: List[Tuple[Tuple[int, int], str]] = []
    for i, m in enumerate(matches):
        y = int(m.group("y") or m.group("y2") or m.group("y3"))
        s = int(m.group("s") or m.group("s2") or m.group("s3"))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        if i == 0:
            prefix = text[: m.start()]
            if prefix.strip():
                chunk = prefix + "\n" + chunk
        blocks.append(((y, s), chunk))
    return blocks


def parse_certificate_section(text: str) -> List[ParsedCertificate]:
    certs: List[ParsedCertificate] = []
    lines = text.splitlines()
    in_cert_block = False
    for ln in lines:
        seg = ln.strip()
        if should_skip_parse_line(seg):
            continue
        gc = parse_glued_cert_line(seg)
        if gc:
            certs.append(gc)
            continue
        if re.search(r"자격증\s*취득|자격\s*인정|국가\s*자격", seg):
            in_cert_block = True
            continue
        if in_cert_block:
            if not seg:
                continue
            if _SEM_HEAD.search(seg) or ("전필" in seg and "합계" in seg and "교양" in seg):
                in_cert_block = False
                continue
            gcb = parse_glued_cert_line(seg)
            if gcb:
                certs.append(gcb)
                continue
            pl = parse_certificate_line_loose(seg)
            if pl:
                certs.append(pl)
            else:
                mc = parse_munged_certificate_segment(seg)
                if mc:
                    certs.append(mc)
                else:
                    slb = parse_certificate_standalone_line(seg)
                    if slb:
                        certs.append(slb)
            continue
        nos = remove_all_inline_ws(seg)
        ec = extract_cert_from_nospace(nos)
        if ec:
            certs.append(ec)
            continue
        if "학점인정" in nos and re.search(r"[가-힣]", seg):
            # (경기대) 학점인정 필요 ○○학3 은 수업 — 여기서 자격증으로 오인하면 안 됨
            if not _line_looks_like_institution_course_after_strip(seg):
                mc = parse_munged_certificate_segment(seg)
                if mc:
                    certs.append(mc)
                    continue
        sl = parse_certificate_standalone_line(seg)
        if sl:
            certs.append(sl)

    seen_k: set[Tuple[str, float, str]] = set()
    dedup: List[ParsedCertificate] = []
    for c in certs:
        k = (c.name, c.credits, c.apply_kind or "")
        if k not in seen_k:
            seen_k.add(k)
            dedup.append(c)
    return dedup


def parse_certificate_line_loose(ln: str) -> Optional[ParsedCertificate]:
    m = _CERT_LINE.match(ln)
    if not m:
        return None
    cred = float(m.group("cred"))
    if cred >= 100:
        return None
    name = m.group("name").strip()
    app = (m.group("app") or "").strip() or None
    if len(name) < 2:
        return None
    return ParsedCertificate(name=name, credits=cred, apply_kind=app)


_DOKHAK_DEGREE_LINE = re.compile(r"독학학위제\s*\d*\s*단계\s*(.+?)\s+(\d+(?:\.\d+)?)\s*학점", re.I)


def parse_dokhak_course_line(ln: str, year: int, sem: int) -> Optional[ParsedCourse]:
    """독학학위제·취득예정n과목명학점 형 — 교양 집계용."""
    raw = ln.strip()
    if not raw:
        return None
    if "독학학위제" in raw or "독학사" in raw:
        m = _DOKHAK_DEGREE_LINE.search(raw)
        if m:
            name = unicodedata.normalize("NFKC", m.group(1).strip())
            cr = float(m.group(2))
            if 0 < cr <= 30 and len(name) >= 1:
                return ParsedCourse(
                    name_raw=name,
                    user_category=DB_CAT_LIB,
                    credits=cr,
                    year=year,
                    semester=sem,
                    institution="독학사",
                    learning_mode="독학사",
                    is_outsource_other_institution=False,
                )
    nos = remove_all_inline_ws(raw)
    if nos.startswith("취득예정"):
        m2 = re.match(r"^취득예정\d+(.+?)(\d{1,2})$", nos)
        if m2:
            name = unicodedata.normalize("NFKC", m2.group(1).strip())
            cr = int(m2.group(2))
            if 1 <= cr <= 30 and len(name) >= 1 and re.search(r"[가-힣A-Za-z]", name):
                return ParsedCourse(
                    name_raw=name,
                    user_category=DB_CAT_LIB,
                    credits=float(cr),
                    year=year,
                    semester=sem,
                    institution="독학사",
                    learning_mode="독학사",
                    is_outsource_other_institution=False,
                )
    return None


def parse_learning_plan_text(text: str) -> ParsedPlan:
    plan = ParsedPlan()
    plan.input_text = text
    if not text.strip():
        plan.parse_warnings.append("입력 텍스트가 비어 있습니다.")
        return plan

    deg_kind: DegreeKind = "미상"
    for pat, label in _DEGREE_PATTERNS:
        if pat.search(text):
            deg_kind = label  # type: ignore
            break

    mq = ""
    mm = _MAJOR_HINT.search(text)
    if mm:
        mq = mm.group(1).strip()
    if not mq:
        mq = detect_major_keyword_from_degree_text(text)
    if not mq:
        head = "\n".join(text.splitlines()[:30])
        m2 = re.search(
            r"(컴퓨터공학|컴공|경영학|경영|체육학|사회복지학|인공지능학|관광경영|정보통신|정보처리|레저스포츠)\s*(전공)?",
            head,
        )
        if m2:
            mq = m2.group(0).strip()
            if mq.startswith("컴공"):
                mq = "컴퓨터공학"

    # IMPORTANT: 입력 본문 전체(과목 목록)를 다시 스캔해 전공을 덮어쓰지 않는다.
    # 희망전공/학위 헤더에서 찾은 mq를 우선 신뢰한다.

    plan.expected_graduation, plan.expected_graduation_explicit = parse_graduation_hint(text)

    for ln in text.splitlines():
        if should_skip_parse_line(ln):
            continue
        summ = parse_compact_summary_line(ln)
        if summ:
            plan.summary_row = summ
            break

    deg_disp_bits = [mq or "미입력"] if mq else ["미입력"]
    if deg_kind != "미상":
        deg_disp_bits.append(str(deg_kind))
    plan.degree_display = " ".join(deg_disp_bits)
    plan.degree_kind = deg_kind
    plan.major_query = mq
    plan.raw_summary = f"추정 학위: {deg_kind} | 전공 키워드: {mq or '(미식별)'}"

    plan.certificates.extend(parse_certificate_section(text))

    blocks = split_semester_blocks(text)

    for (y_def, s_def), chunk in blocks:
        cur_y, cur_s = y_def, s_def
        cert_follow = False
        for line in chunk.splitlines():
            ln = line.strip()
            if not ln:
                cert_follow = False
                continue

            summ = parse_compact_summary_line(ln)
            if summ:
                plan.summary_row = summ
                continue

            if any(kw in ln for kw in _EXCEL_PARSE_JUNK_LINE_KEYWORDS):
                continue

            if should_skip_parse_line(ln):
                continue

            if re.search(r"자격증\s*취득|국가\s*자격", ln):
                cert_follow = True
                continue

            nos_line = remove_all_inline_ws(ln)

            sem_in_line = _SEM_HEAD.search(ln)
            if sem_in_line:
                cert_follow = False
                cur_y = int(sem_in_line.group("y") or sem_in_line.group("y2") or sem_in_line.group("y3"))
                cur_s = int(sem_in_line.group("s") or sem_in_line.group("s2") or sem_in_line.group("s3"))
            else:
                smh = _SEM_NOSPACE_HEAD.match(nos_line)
                if smh:
                    cert_follow = False
                    cur_y = int(smh.group(1) or smh.group(3))
                    cur_s = int(smh.group(2) or smh.group(4))

            # 자격증 전용 줄·구간: 학기 수강 목록(Rule2)에 절대 넣지 않음 (총학점은 parse_certificate_section·과목 합산)
            if cert_follow:
                continue
            dk_course = parse_dokhak_course_line(ln, cur_y, cur_s)
            if dk_course:
                plan.courses.append(dk_course)
                continue
            ec_line = extract_cert_from_nospace(nos_line)
            if ec_line:
                plan.certificates.append(ec_line)
                continue
            # 주의: 여기서 학점인정필요 줄을 스킵하면 (경기대) 학점인정 필요 스포츠심리학3 처럼
            # 접두가 붙은 '수업' 줄이 전처리도 못 하고 증발함 → 스킵 금지
            pl_line = parse_certificate_line_loose(ln)
            if pl_line:
                plan.certificates.append(pl_line)
                continue
            sl_line = parse_certificate_standalone_line(ln)
            if sl_line:
                plan.certificates.append(sl_line)
                continue

            if "독학사" in ln:
                continue

            clean = preprocess_excel_paste_line_for_course(ln)
            if not clean or clean.startswith("취득예정"):
                continue

            for name_raw, credits, ucat, inst_scope in extract_course_invincible_chunks(clean):
                if is_phantom_parsed_course_name(name_raw):
                    continue
                line_inst = _institution_from_raw_line(ln)
                if inst_scope == "타기관":
                    final_inst: Optional[str] = "타기관"
                    is_os = True
                elif inst_scope == "본원":
                    final_inst = line_inst or "본원"
                    is_os = is_outsource_course_name(name_raw)
                else:
                    final_inst = line_inst or "본원"
                    is_os = is_outsource_course_name(name_raw)
                plan.courses.append(
                    ParsedCourse(
                        name_raw=name_raw,
                        user_category=ucat,
                        credits=float(credits),
                        year=cur_y,
                        semester=cur_s,
                        institution=final_inst,
                        learning_mode="수업",
                        is_outsource_other_institution=is_os,
                    )
                )

    # 자격증 중복 제거
    _sk: set[Tuple[str, float, str]] = set()
    _uq: List[ParsedCertificate] = []
    for c in plan.certificates:
        kk = (c.name, c.credits, c.apply_kind or "")
        if kk not in _sk:
            _sk.add(kk)
            _uq.append(c)
    plan.certificates = _uq

    assign_certificate_bucket_labels(plan, plan.major_query or "")

    if not plan.courses:
        plan.parse_warnings.append(
            "수강 과목을 자동 인식하지 못했습니다. 학기 줄(예: 2026년 1학기)과 함께 "
            "`경영학개론3` 또는 `수강예정2026-05-14경영학개론3` 형태로 적혀 있는지 확인해 보세요."
        )

    return plan


# ---------------------------------------------------------------------------
# 검증 규칙
# ---------------------------------------------------------------------------

def _ko_major_cat(cat: str) -> str:
    if cat == DB_CAT_REQ:
        return "전공필수(전필)"
    if cat == DB_CAT_ELEC:
        return "전공선택(전선)"
    if cat == DB_CAT_LIB:
        return "교양"
    if cat == DB_CAT_GEN:
        return "일반"
    return cat


def rule1_classification(
    df_major: pd.DataFrame,
    plan: ParsedPlan,
    major_label: str,
    degree_kind: DegreeKind,
) -> CheckResult:
    issues: List[str] = []
    guides: List[str] = []
    if df_major.empty:
        return CheckResult(
            "R1",
            "과목 분류 팩트체크 (표준교육과정 대조)",
            True,
            "선택된 표준교육과정이 없어 이 검사를 건너뜁니다.",
        )

    db_names = df_major["course_name"].tolist()
    for c in plan.courses:
        if c.learning_mode != "수업":
            continue
        match = best_course_match(course_name_for_db_match(c.name_raw), db_names)
        if not match:
            continue  # 교양·타 영역은 DB 에 없을 수 있음
        row = df_major[df_major["course_name"] == match].iloc[0]
        db_cat = str(row["category"])
        if c.user_category is None:
            guides.append(f"[{c.name_raw}] 에 사용자 적 재구분(전필/전선/교양)이 없습니다. 설계표에 명시하세요.")
            continue
        if db_cat != c.user_category:
            # 요약 브리핑에서 전필로 자동 상향·경고 처리 — R1에서는 중복 FAIL 방지
            if db_cat == DB_CAT_REQ and c.user_category in (DB_CAT_GEN, DB_CAT_LIB):
                continue
            if "사회복지학개론" in _course_name_normalized_compact(c.name_raw) and db_cat == DB_CAT_REQ:
                continue
            if c.user_category == DB_CAT_REQ:
                continue
            dk = degree_kind if degree_kind != "미상" else "해당 학위"
            issues.append(
                f"{c.year}년 {c.semester}학기 [{c.name_raw}] 은(는) **{dk}·{major_label}** 기준으로 표준교육과정에서는 "
                f"{_ko_major_cat(db_cat)} 과목인데, 설계표에는 {_ko_major_cat(c.user_category)} 로 적혀 있습니다."
            )
            guides.append(
                f"'{match}' 과목 구분을 표준교육과정과 동일하게 {_ko_major_cat(db_cat)} 으로 수정하세요."
            )

    ok = len(issues) == 0
    detail = "모든 DB 등재 과목의 구분이 표준교육과정과 일치합니다." if ok else "\n".join(issues)
    return CheckResult("R1", "과목 분류 팩트체크 (표준교육과정 대조)", ok, detail, guides)


def rule2_semester_year_limits(plan: ParsedPlan) -> CheckResult:
    issues: List[str] = []
    guides: List[str] = []

    sem_keyed: Dict[Tuple[int, int], List[ParsedCourse]] = {}
    for c in plan.courses:
        if c.learning_mode not in ("수업", "독학사"):
            continue
        sem_keyed.setdefault((c.year, c.semester), []).append(c)

    for (y, sm), lst in sorted(sem_keyed.items()):
        cr = sum(x.credits for x in lst)
        cnt = len(lst)
        if cr > SEM_CREDIT_CAP:
            issues.append(f"{y}년 {sm}학기 이수 학점 {cr:.1f}학점 — 상한 {SEM_CREDIT_CAP}학점 초과.")
            guides.append("해당 학기 일부 과목을 다음 학기로 이동하거나 학점을 재확인하세요.")
        if cnt > SEM_COURSE_CAP:
            issues.append(f"{y}년 {sm}학기 수강 과목 수 {cnt}개 — 상한 {SEM_COURSE_CAP}과목 초과.")

    year_keyed: Dict[int, List[ParsedCourse]] = {}
    for c in plan.courses:
        if c.learning_mode not in ("수업", "독학사"):
            continue
        year_keyed.setdefault(c.year, []).append(c)

    for y, lst in sorted(year_keyed.items()):
        cr = sum(x.credits for x in lst)
        cnt = len(lst)
        if cr > YEAR_CREDIT_CAP:
            issues.append(f"{y}년도 연간 이수 학점 {cr:.1f}학점 — 상한 {YEAR_CREDIT_CAP}학점 초과.")
            guides.append("연간 학점 배분을 조정하거나 이수 연도를 나누세요.")
        if cnt > YEAR_COURSE_CAP:
            issues.append(f"{y}년도 연간 수강 과목 수 {cnt}개 — 상한 {YEAR_COURSE_CAP}과목 초과.")

    ok = len(issues) == 0
    detail = "학기·연간 수강 상한을 만족합니다." if ok else "\n".join(issues)
    return CheckResult("R2", "연간/학기 수강제한 (24·42학점, 8·14과목)", ok, detail, guides)


def rule3_min_class_credits(plan: ParsedPlan) -> CheckResult:
    class_cr = sum(
        c.credits for c in plan.courses if c.learning_mode == "수업"
    )
    ok = class_cr >= CLASS_CREDIT_MIN
    detail = f"자격증·독학사를 제외한 수업 이수 합계: {class_cr:.1f}학점 (기준 {CLASS_CREDIT_MIN}학점 이상)."
    guides: List[str] = []
    if not ok:
        guides.append("독학사·자격증만으로는 안 되며, 반드시 정규 수업 이수 학점을 확보해야 합니다.")
    return CheckResult("R3", "의무 학점 (수업 18학점 이상)", ok, detail, guides)


def rule4_institution_cap(plan: ParsedPlan, kind: DegreeKind) -> CheckResult:
    default_inst = "단일기관(미기재·본원)"
    buckets: Dict[str, float] = {}
    for c in plan.courses:
        if c.learning_mode != "수업":
            continue
        if is_outsource_course_name(c.name_raw):
            continue
        if (c.institution or "") == "타기관":
            continue
        if c.is_outsource_other_institution:
            continue
        inst = c.institution or default_inst
        if inst == "본원":
            inst = default_inst
        buckets[inst] = buckets.get(inst, 0.0) + c.credits

    cap = INST_CAP_BA if kind in ("학사", "타전공학사") else INST_CAP_AA if kind in ("전문학사", "타전공전문학사") else INST_CAP_BA

    issues: List[str] = []
    guides: List[str] = []
    for inst, total in buckets.items():
        if total > cap:
            issues.append(f"[{inst}] 에서의 이수 학점 {total:.1f}학점이 동일 기관 상한 {cap}학점을 초과합니다.")
            guides.append("분산 이수 또는 기관별 상한 규정을 확인하세요. (학사 105 / 전문학사 60)")

    # 기관 미기재가 여러 건이면 사용자에게 경고
    if len(buckets) > 1 and default_inst in buckets:
        guides.append("일부 행에 교육기관명이 없습니다. 표에 [기관명] 또는 괄호로 기관을 적어 구분하세요.")

    ok = len(issues) == 0
    detail = "동일 기관 이수 한도를 초과하지 않습니다." if ok else "\n".join(issues)
    return CheckResult("R4", "1개 교육기관 제한 (학사 105 / 전문학사 60)", ok, detail, guides)


def _df_major_curriculum_is_computer_engineering(df_major: pd.DataFrame) -> bool:
    """선택된 표준교육과정이 컴퓨터공학 전공인지 (전적대·DB 미매칭 과목 휴리스틱용)."""
    if df_major.empty:
        return False
    parts: List[str] = []
    if "major_title" in df_major.columns:
        parts.extend(df_major["major_title"].astype(str).unique().tolist())
    if "file_stem" in df_major.columns:
        parts.extend(df_major["file_stem"].astype(str).unique().tolist())
    blob = unicodedata.normalize("NFKC", " ".join(parts)).replace(" ", "").lower()
    return "컴퓨터공학" in blob or ("컴퓨터" in blob and "공학" in blob)


def _compeng_it_practice_elec_hint(clean_course_name: str) -> bool:
    """컴공 전선으로 보는 과목명: IT + (기술|실무) 등 (예: IT기술과실무용어)."""
    nm = unicodedata.normalize("NFKC", (clean_course_name or "")).strip()
    if not nm:
        return False
    collapsed = re.sub(r"\s+", "", nm)
    if not re.search(r"(?i)it", collapsed):
        return False
    return ("기술" in nm) or ("실무" in nm)


def _compeng_it_practice_elec_course(c: ParsedCourse, df_major: pd.DataFrame) -> bool:
    if c.learning_mode != "수업" or c.user_category == DB_CAT_GEN:
        return False
    if not _df_major_curriculum_is_computer_engineering(df_major):
        return False
    return _compeng_it_practice_elec_hint(course_name_for_db_match(c.name_raw))


def _df_major_curriculum_is_social_welfare(df_major: pd.DataFrame) -> bool:
    if df_major.empty:
        return False
    parts: List[str] = []
    if "major_title" in df_major.columns:
        parts.extend(df_major["major_title"].astype(str).unique().tolist())
    if "file_stem" in df_major.columns:
        parts.extend(df_major["file_stem"].astype(str).unique().tolist())
    blob = unicodedata.normalize("NFKC", " ".join(parts)).replace(" ", "")
    return "사회복지" in blob


def _db_course_match_name(c: ParsedCourse, df_major: pd.DataFrame) -> Optional[str]:
    if df_major.empty:
        return None
    db_names = df_major["course_name"].tolist()
    return best_course_match(course_name_for_db_match(c.name_raw), db_names)


def _major_req_force_correct_warning(course_display_name: str) -> str:
    return (
        f"⚠️ 행정 오류 정정: [{course_display_name}]은(는) 전공필수 과목이므로 "
        "전필로 자동 상향 조정되었습니다."
    )


def _course_name_normalized_compact(name_raw: str) -> str:
    nm = unicodedata.normalize("NFKC", course_name_for_db_match(name_raw))
    return re.sub(r"\s+", "", nm).replace("\u3000", "")


def _course_credit_bucket_meta(c: ParsedCourse, df_major: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """3단계 스마트 구분: 법정·전필명칭 우선 → DB 전필 교정(+경고) → 설계표 전필 존중 → 나머지."""
    if c.learning_mode == "독학사":
        return DB_CAT_LIB, None
    if c.learning_mode != "수업":
        return DB_CAT_LIB, None

    nm_compact = _course_name_normalized_compact(c.name_raw)
    # 사회복지학 학사 법정 기초 과목 등: 과목명에 '사회복지학개론' 포함 시 무조건 전필
    if "사회복지학개론" in nm_compact:
        return DB_CAT_REQ, None
    # '재수강 사회학개론' 누락 방지: 표기 변형·전공선택 칸 누락과 무관하게 전공선택으로 반영
    # (사회복지 설계표에서 3학점 누락이 빈번하게 발생하는 케이스)
    if "사회학개론" in nm_compact:
        return DB_CAT_ELEC, None

    match_name = _db_course_match_name(c, df_major)
    db_cat: Optional[str] = None
    if match_name is not None:
        db_cat = str(df_major[df_major["course_name"] == match_name].iloc[0]["category"])

    uc = c.user_category

    # 1순위: 표준교육과정 전필(DB)인데 일반·교양으로 적힘 → 전필 강제 + 경고
    if match_name is not None and db_cat == DB_CAT_REQ and uc in (DB_CAT_GEN, DB_CAT_LIB):
        return DB_CAT_REQ, _major_req_force_correct_warning(c.name_raw)

    # 설계표 '전필' 칸 입력은 표준 미등재·DB 구분과 무관하게 전공필수로 인정
    if uc == DB_CAT_REQ:
        return DB_CAT_REQ, None

    # 2순위: DB 미등재 + 사용자가 전선 칸 → 전공선택
    if match_name is None and uc == DB_CAT_ELEC:
        return DB_CAT_ELEC, None

    # 3순위: 사용자 명시 일반(DB 전필 교정 가능 케이스는 위에서 소진)
    if uc == DB_CAT_GEN:
        return DB_CAT_GEN, None

    if match_name is not None and db_cat in (DB_CAT_REQ, DB_CAT_ELEC, DB_CAT_LIB):
        return db_cat, None

    if _compeng_it_practice_elec_course(c, df_major):
        return DB_CAT_ELEC, None

    if match_name is None:
        return DB_CAT_LIB, None

    if uc in (DB_CAT_REQ, DB_CAT_ELEC, DB_CAT_LIB, DB_CAT_GEN):
        return uc, None
    return DB_CAT_LIB, None


def _course_credit_bucket(c: ParsedCourse, df_major: pd.DataFrame) -> str:
    """수업 한 건 최종 구분 전필/전선/교양/일반 (_course_credit_bucket_meta 래핑)."""
    b, _ = _course_credit_bucket_meta(c, df_major)
    return b


def collect_major_req_override_warnings(plan: ParsedPlan, df_major: pd.DataFrame) -> List[str]:
    """1순위 전필 강제 교정 시 사용자 알림 목록."""
    out: List[str] = []
    for c in plan.courses:
        if c.learning_mode != "수업":
            continue
        _, w = _course_credit_bucket_meta(c, df_major)
        if w:
            out.append(w)
    return out


_CERT_MAJOR_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "체육학": ("스포츠", "체육", "건강", "무도"),
    "경영학": ("경영", "회계", "재경", "유통", "cs", "매경", "테셋"),
    "사회복지학": ("사회복지",),
    "컴퓨터공학": (
        "정보처리",
        "네트워크",
        "데이터베이스",
        "데이터",
        "알고리즘",
        "운영체제",
        "자료구조",
        "컴퓨터구조",
        "컴퓨터",
        "c언어",
        "c프로그래밍",
        "프로그래밍",
        "소프트웨어",
    ),
}


def _major_family_key(plan: ParsedPlan, major_label: str) -> str:
    """목표 전공 키(체육학·경영학·사회복지학 등) 추정."""
    ck = resolve_major_canon_from_plan(plan)
    if ck:
        return ck
    for src in (
        plan.major_query,
        plan.degree_display,
        plan.input_text,
        major_label,
        plan.raw_summary,
    ):
        k = detect_major_keyword_from_degree_text(str(src or ""))
        if k:
            return k
    blob = normalize_text(
        " ".join([str(plan.major_query or ""), str(plan.degree_display or ""), str(major_label or "")])
    )
    if "컴퓨터공학" in blob or "컴공" in blob or "컴퓨터" in blob:
        return "컴퓨터공학"
    if "사회복지" in blob:
        return "사회복지학"
    if "체육" in blob:
        return "체육학"
    if "경영" in blob:
        return "경영학"
    if "인공지능" in blob:
        return "인공지능학"
    if "관광경영" in blob:
        return "관광경영학"
    return ""


def _certificate_credit_bucket(cert: ParsedCertificate, plan: ParsedPlan, major_label: str) -> str:
    """자격증 → 전필(전공) 합산 / 교양 / 일반. 목표 전공 키워드 매칭 시 전필."""
    ak = (cert.apply_kind or "").lower()
    if "교양" in ak:
        return DB_CAT_LIB
    if "전공" in ak or "전필" in ak or "전선" in ak:
        return DB_CAT_REQ
    fam = _major_family_key(plan, major_label)
    nm_nfkc = unicodedata.normalize("NFKC", cert.name or "")
    nm_low = nm_nfkc.casefold()
    kws = _CERT_MAJOR_KEYWORDS.get(fam, ())
    if fam and kws:
        for kw in kws:
            if kw.isascii():
                if kw.casefold() in nm_low:
                    return DB_CAT_REQ
            elif kw in nm_nfkc:
                return DB_CAT_REQ
    return DB_CAT_GEN


def assign_certificate_bucket_labels(plan: ParsedPlan, major_label: str = "") -> None:
    """자격증별 집계 구분(전필/교양/일반)을 명시적으로 채움 — 비전공 자격증은 일반."""
    ml = major_label or resolve_major_canon_from_plan(plan) or plan.major_query or ""
    for c in plan.certificates:
        b = _certificate_credit_bucket(c, plan, ml)
        if b == DB_CAT_REQ:
            c.구분 = "전필"
        elif b == DB_CAT_LIB:
            c.구분 = "교양"
        else:
            c.구분 = "일반"


def _required_req_baseline_credits(df_major: pd.DataFrame) -> float:
    """표준교육과정 전공필수 과목 수 × 3학점 (기준 전필 요구 학점)."""
    if df_major.empty:
        return 0.0
    n = len(df_major[df_major["category"] == DB_CAT_REQ]["course_name"].unique())
    return float(n * 3)


def _class_req_credits_on_plan(plan: ParsedPlan, df_major: pd.DataFrame) -> float:
    """수업 중 DB 전공필수로 매칭된 과목 학점 합."""
    if df_major.empty:
        return 0.0
    req_names = set(df_major[df_major["category"] == DB_CAT_REQ]["course_name"].unique())
    db_names = df_major["course_name"].tolist()
    s = 0.0
    for c in plan.courses:
        if c.learning_mode != "수업":
            continue
        m = best_course_match(course_name_for_db_match(c.name_raw), db_names)
        if m and m in req_names:
            s += c.credits
    return s


def _cert_req_credits_sum(plan: ParsedPlan, major_label: str) -> float:
    """전공 자격증(전필 분류) 학점 합."""
    t = 0.0
    for cert in plan.certificates:
        if _certificate_credit_bucket(cert, plan, major_label) == DB_CAT_REQ:
            t += cert.credits
    return t


def _aggregate_by_category(
    plan: ParsedPlan, df_major: pd.DataFrame, major_label: str = ""
) -> Tuple[float, float, float, float]:
    """전공/교양/일반 누적 (과목 구분은 `_course_credit_bucket_meta` 3단계 규칙).

    - 타교육기관 과목도 총 학점에는 반드시 포함한다.
    - 표준 외 전선·전필 표기, DB 전필 교정, 컴공 IT 휴리스틱 등은 같은 메타 로직을 따른다.
    - 자격증은 apply_kind 우선, 미기재 시 이름 힌트로 전공/일반을 추정한다.
    - 총점(total)은 반드시 전공+교양+일반으로 산출한다.
    """
    major_cr = 0.0
    lib_cr = 0.0
    gen_cr = 0.0

    for c in plan.courses:
        if c.learning_mode not in ("수업", "독학사"):
            continue
        b = _course_credit_bucket(c, df_major)
        if b in (DB_CAT_REQ, DB_CAT_ELEC):
            major_cr += c.credits
        elif b == DB_CAT_LIB:
            lib_cr += c.credits
        else:
            gen_cr += c.credits

    for cert in plan.certificates:
        cb = _certificate_credit_bucket(cert, plan, major_label)
        if cb in (DB_CAT_REQ, DB_CAT_ELEC):
            major_cr += cert.credits
        elif cb == DB_CAT_LIB:
            lib_cr += cert.credits
        else:
            gen_cr += cert.credits

    total = major_cr + lib_cr + gen_cr
    return total, major_cr, lib_cr, gen_cr


def _summary_major_req_elec(plan: ParsedPlan) -> Tuple[Optional[float], Optional[float]]:
    """설계표 하단 요약행(전필/전선) 추출값."""
    if not plan.summary_row:
        return None, None
    req = plan.summary_row.get("전필")
    elec = plan.summary_row.get("전선")
    try:
        req_f = float(req) if req is not None else None
    except (TypeError, ValueError):
        req_f = None
    try:
        elec_f = float(elec) if elec is not None else None
    except (TypeError, ValueError):
        elec_f = None
    return req_f, elec_f


def _effective_major_credits(plan: ParsedPlan, computed_major: float) -> float:
    """전공 학점은 파싱 합계와 요약행(전필+전선) 중 큰 값을 사용."""
    req_s, elec_s = _summary_major_req_elec(plan)
    if req_s is None or elec_s is None:
        return computed_major
    return max(computed_major, req_s + elec_s)


def _summary_total(plan: ParsedPlan) -> Optional[float]:
    """설계표 하단 요약행(합계) 추출값."""
    if not plan.summary_row:
        return None
    try:
        v = plan.summary_row.get("합계")
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _effective_total_credits(plan: ParsedPlan, computed_total: float) -> float:
    """총 학점은 파싱 합계와 설계표 하단 합계 중 큰 값을 사용.

    '학점인정 필요' 전적대/기이수 학점이 표 형태로 들어와 파싱에서 누락되더라도,
    설계표가 명시한 총 합계를 기준으로 검증/브리핑이 흔들리지 않게 한다.
    """
    sr = _summary_total(plan)
    if sr is None:
        return computed_total
    return max(computed_total, sr)


def rule5_degree_cutoff(
    plan: ParsedPlan, df_major: pd.DataFrame, kind: DegreeKind, major_label: str = ""
) -> CheckResult:
    total, major_cr, lib_cr, gen_cr = _aggregate_by_category(plan, df_major, major_label)
    major_eval = _effective_major_credits(plan, major_cr)
    total_eval = _effective_total_credits(plan, total)
    issues: List[str] = []
    guides: List[str] = []
    is_tajeongong = _is_tajeongong(plan, kind)

    if is_tajeongong:
        is_aa = (kind in ("전문학사", "타전공전문학사")) or ("전문학사" in (plan.degree_display or ""))
        # 타전공 강제 오버라이드
        target_total = 36.0 if is_aa else 48.0
        target_major = 36.0 if is_aa else 48.0
        target_elective = 0.0
        target_general = 0.0
        _ = (target_total, target_elective, target_general)
        ok = major_eval >= target_major
        detail = f"타전공 이수 요건: 전공 {major_eval:.1f} / {target_major:.1f}"
        if not ok:
            detail = f"타전공 이수 요건 미달: 전공 {major_eval:.1f} / {target_major:.1f}"
        guides.append("타전공은 총점/교양 요건을 면제하고 전공 이수학점(48/36)만 적용합니다.")
        return CheckResult("R5", "학위 요건 커트라인", ok, detail, guides)

    if kind == "학사":
        if total_eval < 140:
            issues.append(f"총 이수 {total_eval:.1f}학점 — 학사 기준 140학점 미만입니다.")
        if major_eval < 60:
            issues.append(f"전공 영역 합산 {major_eval:.1f}학점 — 학사 기준 전공 60학점 미만입니다.")
        if lib_cr < 30:
            issues.append(f"교양 인정 {lib_cr:.1f}학점 — 학사 기준 교양 30학점 미만입니다.")
    elif kind == "전문학사":
        if total_eval < 80:
            issues.append(f"총 이수 {total_eval:.1f}학점 — 전문학사 기준 80학점 미만입니다.")
        if major_eval < 45:
            issues.append(f"전공 영역 합산 {major_eval:.1f}학점 — 전문학사 기준 전공 45학점 미만입니다.")
        if lib_cr < 15:
            issues.append(f"교양 인정 {lib_cr:.1f}학점 — 전문학사 기준 교양 15학점 미만입니다.")
    elif kind == "타전공학사":
        if major_eval < 48:
            issues.append(f"전공(해당 전공 과목) 합산 {major_eval:.1f}학점 — 타전공 학사 기준 48학점 미만입니다.")
        guides.append("타전공 학사는 전공 최소 48학점 이상이어야 합니다.")
    elif kind == "타전공전문학사":
        if major_eval < 36:
            issues.append(f"전공 합산 {major_eval:.1f}학점 — 타전공 전문학사 기준 36학점 미만입니다.")
    else:
        guides.append("학위(학사/전문학사/타전공 등)를 텍스트 상단에 명시하면 요건 검증이 정확해집니다.")

    ok = len(issues) == 0
    if kind == "타전공학사":
        detail = f"타전공 학사 이수 요건: 전공 {major_eval:.1f} / 48.0학점"
    elif kind == "타전공전문학사":
        detail = f"타전공 전문학사 이수 요건: 전공 {major_eval:.1f} / 36.0학점"
    else:
        detail = (
            f"총 {total_eval:.1f}학점 / 전공 {major_eval:.1f} / 교양 {lib_cr:.1f} / 일반 {gen_cr:.1f} (추정 합산)"
            if ok
            else "\n".join(issues)
        )
    return CheckResult("R5", "학위 요건 커트라인", ok, detail, guides)


def _is_social_welfare_target(plan: ParsedPlan, major_label: str) -> bool:
    """사회복지 전공만 법정 10/7 검증. DB major_title·타 전공 문자열 오염으로 오판하지 않음."""
    del major_label
    return resolve_major_canon_from_plan(plan) == "사회복지학"


def _is_tajeongong(plan: ParsedPlan, kind: DegreeKind) -> bool:
    if kind in ("타전공학사", "타전공전문학사"):
        return True
    blob = " ".join([str(plan.degree_display or ""), str(plan.raw_summary or ""), str(plan.major_query or "")])
    return "타전공" in blob


def _major_req_elec_stats(plan: ParsedPlan, df_major: pd.DataFrame) -> Tuple[int, float, int, float]:
    """전필/전선 과목 수·학점 합 (_course_credit_bucket_meta와 동일 규칙으로 집계)."""
    req_cnt = 0
    req_sum = 0.0
    elec_cnt = 0
    elec_sum = 0.0
    for c in plan.courses:
        if c.learning_mode != "수업":
            continue
        b, _ = _course_credit_bucket_meta(c, df_major)
        if b == DB_CAT_REQ:
            req_cnt += 1
            req_sum += c.credits
        elif b == DB_CAT_ELEC:
            elec_cnt += 1
            elec_sum += c.credits
    return req_cnt, req_sum, elec_cnt, elec_sum


def rule6_major_required(
    plan: ParsedPlan,
    df_major: pd.DataFrame,
    major_label: str,
) -> CheckResult:
    if df_major.empty:
        return CheckResult(
            "R6",
            "전공필수 방어",
            True,
            "표준교육과정이 없어 전공필수 목록을 불러오지 못했습니다.",
        )

    # 사회복지 타깃은 법정 과목수(전필 10/전선 7)를 우선 검증
    if _is_social_welfare_target(plan, major_label):
        req_cnt, req_sum, elec_cnt, elec_sum = _major_req_elec_stats(plan, df_major)
        ok = (req_cnt >= 10 and req_sum >= 30.0 and elec_cnt >= 7 and elec_sum >= 21.0)
        issues: List[str] = []
        guides: List[str] = []
        if req_cnt < 10 or req_sum < 30.0:
            issues.append(
                f"사회복지 전필 {req_cnt}과목/{req_sum:.1f}학점 — 기준 10과목/30학점 미달."
            )
        if elec_cnt < 7 or elec_sum < 21.0:
            issues.append(
                f"사회복지 전선 {elec_cnt}과목/{elec_sum:.1f}학점 — 기준 7과목/21학점 미달."
            )
        if not ok:
            guides.append("사회복지사 2급 기준 전필 10과목·전선 7과목 충족 여부를 설계표 과목수로 재점검하세요.")
        detail = (
            f"사회복지 법정 과목수 검증: 전필 {req_cnt}과목/{req_sum:.1f}학점, "
            f"전선 {elec_cnt}과목/{elec_sum:.1f}학점"
            if ok
            else "\n".join(issues)
        )
        return CheckResult("R6", "전공필수 방어", ok, detail, guides)

    required = df_major[df_major["category"] == DB_CAT_REQ]["course_name"].unique().tolist()
    db_names = df_major["course_name"].tolist()

    taken_names: set[str] = set()
    for c in plan.courses:
        if c.learning_mode != "수업":
            continue
        m = best_course_match(course_name_for_db_match(c.name_raw), db_names)
        if m and m in required:
            taken_names.add(m)

    missing = [r for r in required if r not in taken_names]

    baseline = _required_req_baseline_credits(df_major)
    class_req = _class_req_credits_on_plan(plan, df_major)
    cert_req = _cert_req_credits_sum(plan, major_label)
    combined = class_req + cert_req

    issues: List[str] = []
    guides: List[str] = []

    if baseline <= 0:
        ok = len(missing) == 0
        if ok:
            detail = "표준교육과정에 전공필수 행이 없음 — 과목별 검증 생략."
        else:
            detail = (
                f"{major_label} 전공필수 중 미이수 추정: " + ", ".join(missing[:12])
                + (" 외" if len(missing) > 12 else "")
            )
        return CheckResult("R6", "전공필수 방어", ok, detail, guides)

    ok = combined >= baseline

    if ok:
        if missing:
            detail = "완벽 방어 (자격증 학점 대체 인정 ➔ PASS)"
        else:
            detail = "전공필수 과목을 모두 이수한 것으로 확인됩니다."
        guides.append(
            f"전필 기준 {baseline:.1f}학점 — 수업 전필 {class_req:.1f} + 전공 자격증 {cert_req:.1f} = {combined:.1f}학점."
        )
    else:
        gap = baseline - combined
        issues.append(
            f"전공필수 기준 {baseline:.1f}학점 대비 수업·전공자격증 합계 {combined:.1f}학점 — 부족 {gap:.1f}학점."
        )
        if missing:
            issues.append(
                f"{major_label} 전공필수 중 미이수 추정: " + ", ".join(missing[:12])
                + (" 외" if len(missing) > 12 else "")
            )
        guides.append("전필 과목 이수 또는 전공 관련 자격증 학점으로 기준 학점을 채우세요.")
        detail = "\n".join(issues)

    return CheckResult("R6", "전공필수 방어", ok, detail, guides)


@st.cache_data(show_spinner=False)
def load_curriculum_cached(curriculum_dir_str: str) -> Tuple[pd.DataFrame, List[Dict]]:
    return load_curriculum_database(Path(curriculum_dir_str))


def run_all_checks(
    df_major: pd.DataFrame,
    plan: ParsedPlan,
    major_label: str,
    kind: DegreeKind,
) -> List[CheckResult]:
    return [
        rule1_classification(df_major, plan, major_label, kind),
        rule2_semester_year_limits(plan),
        rule3_min_class_credits(plan),
        rule4_institution_cap(plan, kind),
        rule5_degree_cutoff(plan, df_major, kind, major_label),
        rule6_major_required(plan, df_major, major_label),
    ]


def _major_req_elec_sums(plan: ParsedPlan, df_major: pd.DataFrame) -> Tuple[float, float]:
    """사용자 표기·DB 매칭 기준 전필/전선 학점 합."""
    _, req_sum, _, elec_sum = _major_req_elec_stats(plan, df_major)
    s_req, s_elec = _summary_major_req_elec(plan)
    if s_req is not None and s_elec is not None and (s_req + s_elec) >= (req_sum + elec_sum):
        return s_req, s_elec
    return req_sum, elec_sum


def _r6_defense_line(df_major: pd.DataFrame, plan: ParsedPlan, major_label: str) -> str:
    if _is_social_welfare_target(plan, major_label):
        req_cnt, req_sum, elec_cnt, elec_sum = _major_req_elec_stats(plan, df_major)
        ok_sw = (req_cnt >= 10 and req_sum >= 30.0 and elec_cnt >= 7 and elec_sum >= 21.0)
        suffix = "충족" if ok_sw else "미충족"
        return (
            f"사회복지 기준 전필 {req_cnt}/10과목 ({req_sum:.1f}/30), "
            f"전선 {elec_cnt}/7과목 ({elec_sum:.1f}/21) — {suffix}"
        )
    if df_major.empty:
        return "표준과정 없음 (확인 불가)"
    baseline = _required_req_baseline_credits(df_major)
    class_req = _class_req_credits_on_plan(plan, df_major)
    cert_req = _cert_req_credits_sum(plan, major_label)
    combined = class_req + cert_req
    required = df_major[df_major["category"] == DB_CAT_REQ]["course_name"].unique().tolist()
    db_names = df_major["course_name"].tolist()
    taken: set[str] = set()
    for c in plan.courses:
        if c.learning_mode != "수업":
            continue
        m = best_course_match(course_name_for_db_match(c.name_raw), db_names)
        if m and m in required:
            taken.add(m)
    missing = [r for r in required if r not in taken]
    if baseline <= 0:
        return "전공필수 DB 없음 — 요약 생략"
    if combined >= baseline:
        return (
            f"완벽 방어 (자격증 학점 대체 인정 ➔ PASS) "
            f"— 기준 {baseline:.1f} / 합계 {combined:.1f} (수업전필 {class_req:.1f} + 자격증 {cert_req:.1f})"
        )
    gap = baseline - combined
    tail = ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")
    return f"전필 기준 대비 부족 {gap:.1f}학점 — 미이수 추정: {tail}"


def _expected_grad_display(plan: ParsedPlan) -> str:
    if plan.expected_graduation:
        if getattr(plan, "expected_graduation_explicit", False):
            return f"{plan.expected_graduation} (설계표 기준)"
        return plan.expected_graduation
    for c in plan.courses:
        if c.year and c.semester:
            y = c.year + (1 if c.semester == 2 else 0)
            return f"{y}-02 (자동 추정, 참고용)"
    return "미기재"


def parse_kyonggi_declared_main_credits(blob: str) -> Optional[float]:
    """설계표 하단 '경기대학교 이수학점 NN학점' 등 명시값 추출."""
    if not blob:
        return None
    t = unicodedata.normalize("NFKC", blob)
    m = re.search(r"경기대(?:학교)?\s*이수학점\s*(\d+(?:\.\d+)?)\s*학?", t)
    if m:
        return float(m.group(1))
    return None


def total_gyeonggi_main_campus_credits(plan: ParsedPlan) -> float:
    """본원 경기 과정으로 이수한 **수업** 학점 (명시 타기관·전적대·외주만 제외).

    기관 문자열에 '경기대'·'본원'이 없더라도, `(원격교육원)`만 붙어 집계에서 빠지던 경우를 방지하기 위해
    **`institution == 타기관`인 경우만** 본원에서 제외한다. 설계표에 경기대 이수학점이 숫자로 적혀 있으면 그 값과 파싱 합계 중 큰 값을 사용한다.
    """
    s = 0.0
    for c in plan.courses:
        if c.learning_mode != "수업":
            continue
        if c.is_outsource_other_institution:
            continue
        inst = (c.institution or "").strip()
        if inst == "타기관":
            continue
        s += c.credits
    dec = parse_kyonggi_declared_main_credits(plan.input_text or "")
    if dec is not None and dec > s - 1e-6:
        return dec
    return s


def credit_total_reconciliation_line(plan: ParsedPlan) -> str:
    """수업·독학사 + 자격증 합계 검산 및 설계표 요약 합계 대조."""
    course_cr = sum(
        float(c.credits) for c in plan.courses if c.learning_mode in ("수업", "독학사")
    )
    cert_cr = sum(float(c.credits) for c in plan.certificates)
    tot = course_cr + cert_cr
    parts: List[str] = [
        f"총학점 검산: 수업·독학사 {course_cr:.1f} + 자격증·학점인정 {cert_cr:.1f} = **{tot:.1f}학점**",
        "(전적대·타기관 수업·학점인정 줄은 위 합계에 포함; 제외 처리 없음)",
    ]
    if plan.summary_row and "합계" in plan.summary_row:
        sr = float(plan.summary_row["합계"])
        parts.append(f"설계표 하단 요약 합계 {sr:g}학점과 차이 **{tot - sr:+.1f}학점**")
    return "* " + " ".join(parts)


def _kyonggi_president_credit_threshold(kind: DegreeKind, plan: ParsedPlan) -> float:
    """경기대 원격교육원 기준: 일반 학사·전문학사 84 / 타전공 학사 48 / 타전공 전문학사 36."""
    if kind == "타전공학사":
        return 48.0
    if kind == "타전공전문학사":
        return 36.0
    if kind in ("학사", "전문학사"):
        return 84.0
    if _is_tajeongong(plan, kind):
        dd = str(plan.degree_display or "")
        if "타전공" in dd and "전문학사" in dd:
            return 36.0
        return 48.0
    return 84.0


def _fmt_briefing_credit(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


def kyonggi_president_briefing_line(plan: ParsedPlan, kind: DegreeKind) -> str:
    """경기대학교 총장명의 학위 수여 가능 여부 요약 문구."""
    # 설계표 원문(input_text)에는 '교육부' 등 행정 문구가 많아 오탐됨 → 목표 학위로 파생된 텍스트만 사용
    goal_blob = "".join([str(plan.degree_display or ""), str(plan.major_query or "")])
    if "교육부" in goal_blob or "장관명의" in goal_blob:
        return "NO (학습자 희망: 교육부 장관명의)"
    main_cr = total_gyeonggi_main_campus_credits(plan)
    thresh = _kyonggi_president_credit_threshold(kind, plan)
    main_s = _fmt_briefing_credit(main_cr)
    t_int = int(round(thresh))
    if main_cr >= thresh:
        return (
            f"YES (본원 이수: {main_s} / 기준: {t_int}) ➔ 경기대 총장명의 대상자입니다. 🎉"
        )
    gap = max(0.0, thresh - main_cr)
    gap_n = int(math.ceil(gap - 1e-6)) if gap > 1e-6 else 0
    return (
        f"NO (본원 이수: {main_s} / 기준: {t_int}) ➔ 경기대 총장명의를 위해 본원에서 {gap_n}학점 추가 수강이 필요합니다."
    )


def build_briefing_markdown(
    plan: ParsedPlan,
    df_major: pd.DataFrame,
    _major_label: str,
    kind: DegreeKind,
    results: List[CheckResult],
) -> str:
    rmap = {r.rule_id: r for r in results}
    total, major_cr, lib_cr, gen_cr = _aggregate_by_category(plan, df_major, _major_label)
    total = _effective_total_credits(plan, total)
    major_cr = _effective_major_credits(plan, major_cr)
    req_s, elec_s = _major_req_elec_sums(plan, df_major)
    is_tajeongong = _is_tajeongong(plan, kind)

    if kind == "학사":
        t_tot, t_maj, t_lib = 140.0, 60.0, 30.0
    elif kind == "전문학사":
        t_tot, t_maj, t_lib = 80.0, 45.0, 15.0
    elif kind == "타전공학사":
        t_tot, t_maj, t_lib = 140.0, 48.0, 30.0
    elif kind == "타전공전문학사":
        t_tot, t_maj, t_lib = 80.0, 36.0, 15.0
    else:
        t_tot, t_maj, t_lib = 140.0, 60.0, 30.0

    if is_tajeongong:
        t_maj = 36.0 if (kind in ("전문학사", "타전공전문학사") or "전문학사" in (plan.degree_display or "")) else 48.0
        pf_total = "PASS" if major_cr >= t_maj else "FAIL"
        pf_major = pf_total
        pf_lib = "—"
        tot_line = f"* **타전공 이수 요건:** {major_cr:.1f} / {t_maj:.1f} ➔ **{pf_total}**"
        lib_line = "* **교양 학점 요건:** 타전공(교양 면제)"
    else:
        pf_total = "PASS" if total >= t_tot else "FAIL"
        pf_major = "PASS" if major_cr >= t_maj else "FAIL"
        pf_lib = "PASS" if lib_cr >= t_lib else "FAIL"
        tot_line = (
            f"* **총 학점 요건:** {total:.1f} / {t_tot:.0f} ➔ **{pf_total}** "
            f"(전공 {major_cr:.1f} / 교양 {lib_cr:.1f} / 일반 {gen_cr:.1f})"
        )
        lib_line = f"* **교양 학점 요건:** {lib_cr:.1f} / {t_lib:.0f} ➔ **{pf_lib}**"

    pf_r6 = "PASS" if rmap["R6"].ok else "FAIL"
    admin_ok = rmap["R2"].ok and rmap["R4"].ok and rmap["R3"].ok
    pf_admin = "PASS" if admin_ok else "FAIL"

    deg_show = plan.degree_display or f"{plan.major_query or '미입력'} ({kind})"

    lines = [
        "### 👑 학습설계 팩트 체크 요약",
        f"* **목표 학위:** {deg_show}",
        f"* **예상 취득일:** {_expected_grad_display(plan)}",
        tot_line,
        f"* **전공 학점 요건:** {major_cr:.1f} (전필 {req_s:.1f} / 전선 {elec_s:.1f}) ➔ **{pf_major}**",
        f"* **전공필수 방어:** {_r6_defense_line(df_major, plan, _major_label)} ➔ **{pf_r6}**",
        lib_line,
        f"* **행정 규정 (연간제한·18학점·1개 기관 등):** {'위반 없음' if admin_ok else '일부 위반 가능'} ➔ **{pf_admin}**",
        f"* **경기대학교 총장명의 수여 여부:** {kyonggi_president_briefing_line(plan, kind)}",
        credit_total_reconciliation_line(plan),
    ]
    for adj in collect_major_req_override_warnings(plan, df_major):
        lines.append(adj)
    lines.extend(
        [
            "",
            "*※ 숫자는 붙여넣은 표와 표준교육과정 매칭으로 **추정**한 값입니다. 결재 전 행정실 확인이 필요합니다.*",
        ]
    )
    return "\n".join([ln for ln in lines if ln != ""])


def build_error_report_md(results: List[CheckResult]) -> str:
    blocks: List[str] = []
    for r in results:
        if r.ok:
            continue
        blocks.append(f"- **{r.rule_id} {r.title}**  \n  {r.detail.replace(chr(10), '  \n  ')}")
        for g in r.guides:
            blocks.append(f"  - 수정 제안: {g}")
    if not blocks:
        return ""
    head = "\n\n### 🚨 오류 팩트 체크 및 수정 제안\n\n"
    return head + "\n".join(blocks)


def build_parsed_inventory_dataframe(
    plan: ParsedPlan, df_major: pd.DataFrame, major_label: str = ""
) -> pd.DataFrame:
    """브리핑 아래 디버그용: 파싱 확정 목록 (집계 버킷과 동일한 전필/전선/교양 구분)."""
    cat_label = {
        DB_CAT_REQ: "전필",
        DB_CAT_ELEC: "전선",
        DB_CAT_LIB: "교양",
        DB_CAT_GEN: "일반",
    }
    rows: List[Dict[str, object]] = []
    for c in plan.courses:
        if c.learning_mode not in ("수업", "독학사"):
            continue
        b = _course_credit_bucket(c, df_major)
        nm = c.name_raw
        if c.learning_mode == "독학사":
            nm = f"[독학사] {c.name_raw}"
        rows.append(
            {
                "수강연도-학기": f"{c.year}년 {c.semester}학기",
                "과목명": nm,
                "학점": c.credits,
                "과목구분": cat_label.get(b, "교양"),
                "구분": cat_label.get(b, "교양"),
            }
        )
    ml = major_label or plan.major_query or ""
    for cert in plan.certificates:
        gu = getattr(cert, "구분", "").strip()
        if gu in ("전필", "교양", "일반"):
            clab = f"자격증({gu})"
        else:
            ck = _certificate_credit_bucket(cert, plan, ml)
            if ck == DB_CAT_REQ:
                clab = "자격증(전필)"
            elif ck == DB_CAT_LIB:
                clab = "자격증(교양)"
            else:
                clab = "자격증(일반)"
        rows.append(
            {
                "수강연도-학기": "—",
                "과목명": cert.name,
                "학점": cert.credits,
                "과목구분": clab,
                "구분": gu or "일반",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="학점은행제 학습설계표 검증", layout="wide")
    st.title("학점은행제 학습설계표 자동 검증")
    st.caption("표준교육과정 텍스트와 비교하여 행정 요건을 1차 점검합니다. 최종은 항상 행정실 확인입니다.")

    base = Path(__file__).resolve().parent
    curriculum_dir = base / "curriculums"

    df_all, metas = load_curriculum_cached(str(curriculum_dir))
    if df_all.empty:
        st.error("`curriculums` 폴더에 표준교육과정 `.txt` 파일을 넣어 주세요.")
        st.stop()

    with st.sidebar:
        st.subheader("표준교육과정 DB")
        st.write(f"파일 {len(metas)}개, 과목 행 {len(df_all)}건 적재됨.")
        st.multiselect(
            "포함된 파일",
            options=[Path(m["path"]).name for m in metas],
            default=[Path(m["path"]).name for m in metas],
            disabled=True,
        )

    default_sample = """희망전공: 경영학 학사
졸업예정: 2028-02

2026년 1학기
수강예정2026-05-14경영학개론3
진행예정2026년 2학기 진행예정경영통계학3
생산관리 전필 3

2026년 2학기
회계원리 전선 3

학점인정 필요전산회계1급4

자격증 취득
정보처리기사 14학점 적용구분: 전공일부

전필25전선24교양27일반6합계82
"""

    text = st.text_area(
        "학습설계표 전체 붙여넣기",
        value=default_sample,
        height=320,
        placeholder="포털·엑셀·웹에서 복사한 그대로 붙여 넣으세요. 학기 헤더(예: 2026년 1학기)와 과목 줄을 포함합니다.",
    )

    st.caption("입력 후 **검증 실행**을 눌러 주세요.")
    col_a, col_b = st.columns(2)
    with col_a:
        run = st.button("검증 실행", type="primary")
    with col_b:
        st.write("")

    if run:
        plan = parse_learning_plan_text(text)
        kind: DegreeKind = plan.degree_kind
        sub = pick_curriculum_subset(df_all, metas, plan, kind)
        major_title = ""
        if not sub.empty:
            major_title = str(sub.iloc[0]["major_title"])
        elif metas:
            major_title = metas[0].get("major_title", "")

        major_canon = resolve_major_canon_from_plan(plan)
        label_for_rules = major_canon or plan.major_query or major_title or "선택 전공"

        results = run_all_checks(sub, plan, label_for_rules, kind)

        st.markdown(
            build_briefing_markdown(
                plan,
                sub,
                label_for_rules,
                kind,
                results,
            )
        )

        st.subheader("파싱된 수강·자격 목록")
        inv_df = build_parsed_inventory_dataframe(
            plan, sub, label_for_rules
        )
        if inv_df.empty:
            st.caption("인식된 수업 과목·자격증이 없습니다.")
        else:
            st.dataframe(
                inv_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "수강연도-학기": st.column_config.TextColumn("수강연도-학기", width="small"),
                    "과목명": st.column_config.TextColumn("과목명", width="large"),
                    "학점": st.column_config.NumberColumn("학점", format="%.1f"),
                    "과목구분": st.column_config.TextColumn("과목구분", width="small"),
                },
            )

        err_md = build_error_report_md(results)
        if err_md:
            st.markdown(err_md)

        for w in plan.parse_warnings:
            st.warning(w)

        all_ok = all(r.ok for r in results)
        if all_ok:
            total, major_cr, lib_cr, gen_cr = _aggregate_by_category(
                plan, sub, label_for_rules
            )
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["총 이수(추정)", "전공(추정)", "교양(추정)", "일반(추정)"],
                        y=[total, major_cr, lib_cr, gen_cr],
                        marker_color=["#3366cc", "#109618", "#dc3912", "#ff9900"],
                    )
                ]
            )
            fig.update_layout(height=360, margin=dict(t=40, b=40), yaxis_title="학점")
            st.success("무결점 설계표입니다. 결재를 진행하세요.")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("위 요약의 FAIL 항목을 수정한 뒤 다시 검증하면 됩니다.")

        with st.expander("추출 내역 · 파싱 요약 (표 대조)", expanded=False):
            disp_mt = major_canon or major_title or plan.major_query or ""
            st.markdown(plan.raw_summary + (" — 표준과정: **" + disp_mt + "**" if disp_mt else ""))
            if plan.summary_row:
                st.markdown("**하단 요약행 추출:** " + ", ".join(f"{k}={v:g}" for k, v in plan.summary_row.items()))
            if plan.courses:
                st.dataframe(pd.DataFrame([c.__dict__ for c in plan.courses]), use_container_width=True)
            if plan.certificates:
                st.dataframe(pd.DataFrame([c.__dict__ for c in plan.certificates]), use_container_width=True)

        with st.expander("입력 형식 도움말"):
            st.markdown(
                """
- **붙어 있는 과목:** `수강예정2026-05-14경영학개론3` → 날짜는 무시하고 **경영학개론·3학점**만 추출합니다.
- **학기가 줄 안에 있을 때:** `진행예정2026년 2학기 진행예정경영통계학3` → **2026년 2학기**로 묶습니다.
- **자격증:** `학점인정 필요전산회계1급4` 또는 `자격증 취득` 블록 아래 줄.
- **요약 한 줄:** `전필25전선24교양27일반6합계82` → 과목으로 세지 않고 요약만 해석합니다.
- **연도·행번호는 학점 아님:** `2026`, `05-14`, 세 자리 이상 숫자는 학점으로 쓰지 않습니다.
"""
            )


if __name__ == "__main__":
    main()
