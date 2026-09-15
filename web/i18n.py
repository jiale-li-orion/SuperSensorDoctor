"""Minimal inline i18n for the doctor workstation.

Design: no central key catalog. Templates call ``L('中文', 'English')`` and
client-side code calls ``T('中文', 'English')``, so a translation always sits
next to the markup it belongs to and cannot drift out of sync.

The active language travels in ``?lang=`` or the ``lang`` cookie and is
resolved per request by a Jinja context processor (see ``web/app.py``).

A ``ContextVar`` mirrors the resolved language so environment-level Jinja
filters (registered once, not per request) can localize too.
"""

from contextvars import ContextVar

SUPPORTED = ("zh", "en")
DEFAULT_LANG = "zh"
COOKIE_NAME = "lang"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365

_current: ContextVar[str] = ContextVar("ssd_lang", default=DEFAULT_LANG)


def normalize(lang: str | None) -> str:
    """Coerce an arbitrary value to a supported language tag."""
    if not lang:
        return DEFAULT_LANG
    tag = str(lang).strip().lower().replace("_", "-")
    if tag.startswith("zh"):
        return "zh"
    if tag.startswith("en"):
        return "en"
    return DEFAULT_LANG


def resolve(*candidates: str | None) -> str:
    """First non-empty candidate wins; falls back to the default."""
    for candidate in candidates:
        if candidate:
            return normalize(candidate)
    return DEFAULT_LANG


def set_lang(lang: str) -> str:
    tag = normalize(lang)
    _current.set(tag)
    return tag


def get_lang() -> str:
    return _current.get()


def pick(zh: str, en: str, lang: str | None = None) -> str:
    """Return the string for ``lang`` (defaults to the active language)."""
    tag = normalize(lang) if lang else _current.get()
    return zh if tag == "zh" else en


def translator(lang: str):
    """Build an ``L(zh, en)`` callable bound to one language."""
    tag = normalize(lang)

    def L(zh: str, en: str) -> str:
        return zh if tag == "zh" else en

    return L


def field(obj, name: str, lang: str):
    """Pick ``name`` or its ``<name>_zh`` sibling from a mapping or object.

    Used for data-driven labels that ship both languages side by side
    (e.g. ``task`` / ``task_zh``, ``INDICATOR_COLUMNS`` / ``INDICATOR_COLUMNS_ZH``).
    Falls back to the base field whenever the localized variant is absent.
    """
    def _get(key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    if normalize(lang) == "zh":
        localized = _get(f"{name}_zh")
        if localized is not None:
            return localized
    return _get(name)


def context(request) -> dict:
    """Jinja context processor: resolves the language for this request."""
    query_lang = None
    try:
        query_lang = request.query_params.get("lang")
    except Exception:
        query_lang = None

    cookie_lang = None
    try:
        cookie_lang = request.cookies.get(COOKIE_NAME)
    except Exception:
        cookie_lang = None

    lang = resolve(query_lang, cookie_lang)
    set_lang(lang)
    return {
        "lang": lang,
        "L": translator(lang),
        "F": lambda obj, name: field(obj, name, lang),
        "lang_other": "en" if lang == "zh" else "zh",
    }
