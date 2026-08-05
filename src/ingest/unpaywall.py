"""Unpaywall：按 DOI 兜底查找合法 OA 全文副本。

仅在文献自身未携带 pdf_url 时调用。需要 email 参数。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.config import get_settings
from src.ingest.base import http_get

logger = logging.getLogger(__name__)

UNPAYWALL_ENDPOINT = "https://api.unpaywall.org/v2"


def find_oa_pdf(doi: str) -> Tuple[Optional[str], Optional[str]]:
    """按 DOI 查最佳 OA 副本。

    返回 (pdf_url, license)。找不到返回 (None, None)。
    落盘时应记录 license 以满足版权合规。
    """
    if not doi:
        return None, None
    settings = get_settings()
    resp = http_get(
        f"{UNPAYWALL_ENDPOINT}/{doi}",
        source="unpaywall",
        params={"email": settings.contact_email},
        retries=1,
    )
    if resp is None:
        return None, None
    try:
        data = resp.json()
    except ValueError:
        return None, None

    best = data.get("best_oa_location")
    if not best:
        return None, None
    pdf_url = best.get("url_for_pdf") or best.get("url")
    return pdf_url, best.get("license")
