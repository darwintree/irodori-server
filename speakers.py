"""Shiny Colors 29-character alias resolution and speaker embedding lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class SpeakerEntry:
    speaker_id: str
    ja_full: str
    ja_short: str
    zh_full: str
    zh_short: str
    extra_aliases: tuple[str, ...] = ()


SPEAKER_CATALOG: tuple[SpeakerEntry, ...] = (
    # illumination STARS
    SpeakerEntry("mano", "櫻木真乃", "真乃", "樱木真乃", "真乃", ("桜木真乃", "まの")),
    SpeakerEntry("hiori", "風野灯織", "灯織", "风野灯织", "灯织", ("ひおり",)),
    SpeakerEntry("meguru", "八宮めぐる", "めぐる", "八宫巡", "巡", ("八宮巡",)),
    # L'Antica
    SpeakerEntry("kogane", "月岡恋鐘", "恋鐘", "月冈恋钟", "恋钟", ("こがね",)),
    SpeakerEntry("mamimi", "田中摩美々", "摩美々", "田中摩美美", "摩美美", ("まみみ",)),
    SpeakerEntry("sakuya", "白瀬咲耶", "咲耶", "白濑咲耶", "咲耶", ("さくや",)),
    SpeakerEntry("yuika", "三峰結華", "結華", "三峰结华", "结华", ("ゆいか", "三峰")),
    SpeakerEntry("kiriko", "幽谷霧子", "霧子", "幽谷雾子", "雾子", ("きりこ",)),
    # Houkago Climax Girls
    SpeakerEntry("kaho", "小宮果穂", "果穂", "小宫果穗", "果穗", ("小宮果穗", "かほ")),
    SpeakerEntry("chiyoko", "園田智代子", "智代子", "园田智代子", "智代子", ("ちよこ", "ちょこ")),
    SpeakerEntry("juri", "西城樹里", "樹里", "西城树里", "树里", ("じゅり",)),
    SpeakerEntry("rinze", "杜野凛世", "凛世", "杜野凛世", "凛世", ("りんぜ",)),
    SpeakerEntry("natsuha", "有栖川夏葉", "夏葉", "有栖川夏叶", "夏叶", ("なつは",)),
    # ALSTROEMERIA
    SpeakerEntry("amana", "大崎甘奈", "甘奈", "大崎甘奈", "甘奈", ("あまな",)),
    SpeakerEntry("tenka", "大崎甜花", "甜花", "大崎甜花", "甜花", ("てんか",)),
    SpeakerEntry("chiyuki", "桑山千雪", "千雪", "桑山千雪", "千雪", ("ちゆき",)),
    # Straylight
    SpeakerEntry("asahi", "芹沢あさひ", "あさひ", "芹泽朝日", "朝日", ("芹澤あさひ",)),
    SpeakerEntry("fuyuko", "黛冬優子", "冬優子", "黛冬优子", "冬优子", ("ふゆこ", "ふゆ")),
    SpeakerEntry("mei", "和泉愛依", "愛依", "和泉爱依", "爱依", ("めい",)),
    # noctchill
    SpeakerEntry("toru", "浅倉透", "透", "浅仓透", "透", ("淺倉透", "とおる", "tooru")),
    SpeakerEntry("madoka", "樋口円香", "円香", "樋口圆香", "圆香", ("まどか",)),
    SpeakerEntry("koito", "福丸小糸", "小糸", "福丸小糸", "小糸", ("こいと",)),
    SpeakerEntry("hinana", "市川雛菜", "雛菜", "市川雏菜", "雏菜", ("ひなな",)),
    # SHHis
    SpeakerEntry("nichika", "七草にちか", "にちか", "七草日花", "日花", ()),
    SpeakerEntry("mikoto", "緋田美琴", "美琴", "绯田美琴", "美琴", ("みこと",)),
    # CoMETIK
    SpeakerEntry("luca", "斑鳩ルカ", "ルカ", "斑鸠路加", "路加", ("るか", "ruka")),
    SpeakerEntry("hana", "鈴木羽那", "羽那", "铃木羽那", "羽那", ("はな",)),
    SpeakerEntry("haruki", "郁田はるき", "はるき", "郁田春希", "春希", ()),
    # 283 Pro Staff
    SpeakerEntry("hazuki", "七草はづき", "はづき", "七草叶月", "叶月", ("七草葉月", "葉月")),
)

_WHITESPACE_RE = re.compile(r"[\s\u3000]+")
_UNSAFE_TASK_CHARS_RE = re.compile(r"[/\\:\x00-\x1f\x7f]+")


def _normalize_alias(name: str) -> str:
    return _WHITESPACE_RE.sub("", name.strip()).lower()


def _build_alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in SPEAKER_CATALOG:
        candidates = (
            entry.speaker_id,
            entry.ja_full,
            entry.ja_short,
            entry.zh_full,
            entry.zh_short,
            *entry.extra_aliases,
        )
        for alias in candidates:
            key = _normalize_alias(alias)
            if key and key not in mapping:
                mapping[key] = entry.speaker_id
    return mapping


ALIAS_TO_SPEAKER_ID: dict[str, str] = _build_alias_map()


def resolve_speaker_id(character_name: str) -> str | None:
    """Resolve a character alias (English ID, Japanese/Chinese short or full name) to canonical ID."""
    key = _normalize_alias(str(character_name or ""))
    if not key:
        return None
    return ALIAS_TO_SPEAKER_ID.get(key)


def resolve_speaker_embed(character_name: str, speakers_dir: Path) -> Path | None:
    """Resolve the 49 KB `.speaker.safetensors` path for a character name, or None if unavailable."""
    raw = str(character_name or "").strip()
    if not raw:
        return None

    speaker_id = resolve_speaker_id(raw)
    if speaker_id is not None:
        candidate = speakers_dir / f"{speaker_id}.speaker.safetensors"
        if candidate.is_file():
            return candidate

    # Allow custom `<name>.speaker.safetensors` dropped into speakers_dir without code changes,
    # while rejecting path traversal segments.
    if "/" not in raw and "\\" not in raw and raw not in {".", ".."}:
        direct = speakers_dir / f"{raw}.speaker.safetensors"
        if direct.is_file():
            return direct
    return None


def sanitize_task_prefix(character_name: str) -> str:
    """Produce a filesystem- and R2-key-safe prefix for task_id."""
    speaker_id = resolve_speaker_id(character_name)
    if speaker_id is not None:
        return speaker_id
    cleaned = _UNSAFE_TASK_CHARS_RE.sub("_", str(character_name or "").strip())
    cleaned = cleaned.replace("..", "_").strip("._")
    return cleaned or "tts"
