import html
import re
from requests import Session

DOI_RE = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.I)

def get_paper_batch(session: Session, ids: list[str], fields: str, **kwargs) -> list[dict]:
    params = {'fields': fields, **kwargs}
    body = {'ids': ids}
    with session.post(
        'https://api.semanticscholar.org/graph/v1/paper/batch',
        params=params,
        json=body,
        timeout=30,
    ) as response:
        response.raise_for_status()
        return response.json()

def extract_doi(paper: dict) -> str | None:
    # 1) если DOI уже есть в externalIds
    doi = (paper.get('externalIds') or {}).get('DOI')
    if doi:
        return doi

    # 2) если DOI есть только в disclaimer
    disclaimer = ((paper.get('openAccessPdf') or {}).get('disclaimer') or '')
    m = DOI_RE.search(disclaimer)
    return m.group(1) if m else None

def get_crossref_abstract(session: Session, doi: str, mailto: str | None = None) -> str | None:
    url = f'https://api.crossref.org/works/{doi}'
    params = {'mailto': mailto} if mailto else None

    with session.get(url, params=params, timeout=30) as response:
        if response.status_code != 200:
            return None
        message = response.json().get('message', {})
        abstract = message.get('abstract')
        if not abstract:
            return None

        # Crossref часто возвращает abstract с XML/JATS-тегами
        abstract = html.unescape(abstract)
        abstract = re.sub(r'<[^>]+>', '', abstract).strip()
        return abstract or None

def get_papers_with_abstracts(session: Session, ids: list[str], mailto: str | None = None) -> list[dict]:
    papers = get_paper_batch(
        session,
        ids,
        fields='paperId,title,abstract,externalIds,openAccessPdf'
    )

    for paper in papers:
        if paper.get('abstract'):
            paper['abstract_source'] = 'semantic_scholar'
            continue

        doi = extract_doi(paper)
        paper['doi'] = doi

        if doi:
            fallback_abstract = get_crossref_abstract(session, doi, mailto=mailto)
            if fallback_abstract:
                paper['abstract'] = fallback_abstract
                paper['abstract_source'] = 'crossref'
            else:
                paper['abstract_source'] = None
        else:
            paper['abstract_source'] = None

    return papers

ids = [
    "46566caff9e7aee755440b3e0687f8863c15005c",
    "6b42da07552dd40974f793f8da1ca6521f1e49e8",
    "3c24bf1754c57966005db8f5187fca69df03c518",
    "e9281f322340420a0910eb37108cda1a98f2b73a",
    "3a6270b9ebfe84642b72c289e32f9950c41d8b74",
]

with Session() as session:
    res = get_papers_with_abstracts(session, ids, 'paperId,title,abstract')
    print(res)