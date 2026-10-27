"""Stage, build and check Margam's user manual; never import the numerical engine."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/manual/source'
SITE=ROOT/'build/manual/site'
LINK=re.compile(r'(\]\()([^\s)]+)([^)]*\))')


PRIVATE_PATH=re.compile(r"/(?:home|Users)/[^/\s]+|[A-Za-z]:[\\/]Users[\\/][^\\/\s]+")


def validate_public_text(text,label):
    if PRIVATE_PATH.search(text):raise ValueError(f'Private home-directory path in {label}; use repository-relative example paths.')


def prepare():
    """Keep maintained Markdown/assets in place; rewrite only generated copies."""
    mapping={p:p.relative_to(ROOT/'docs') for p in (ROOT/'docs').rglob('*') if p.is_file()}
    mapping[ROOT/'docs/README.md']=Path('reference-index.md')
    mapping.update({p:Path('assets')/p.relative_to(ROOT/'assets') for p in (ROOT/'assets').rglob('*') if p.is_file()})
    for name in ('README.md','AGENTS.md','THIRD_PARTY.md','LICENSE'):
        p=ROOT/name
        if p.exists():mapping[p]=Path('project')/name
    pending=[p for p in mapping if p.suffix=='.md'];rewritten={}
    while pending:
        source=pending.pop();destination=mapping[source];fence=None;lines=[]
        validate_public_text(source.read_text(),source.relative_to(ROOT))
        for line in source.read_text().splitlines(keepends=True):
            marker=re.match(r'^\s*(`{3,}|~{3,})',line)
            if marker:
                if fence is None:fence=marker[1][0]
                elif marker[1][0]==fence:fence=None
                lines.append(line);continue
            if fence is not None:lines.append(line);continue
            def rewrite(match):
                target=match[2];parts=urlsplit(target)
                if parts.scheme or parts.netloc or not parts.path:return match[0]
                path=(source.parent/unquote(parts.path)).resolve()
                if not path.is_relative_to(ROOT):raise ValueError(f'Local link outside repository: {source}: {target}')
                if not path.is_file():raise ValueError(f'Missing documentation target: {source.relative_to(ROOT)}: {target}')
                if path not in mapping:
                    mapping[path]=Path('source')/path.relative_to(ROOT)
                    if path.suffix=='.md':pending.append(path)
                import posixpath
                relative=posixpath.relpath(mapping[path].as_posix(),destination.parent.as_posix())
                suffix=('?'+parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
                return match[1]+relative+suffix+match[3]
            validate_public_text(line,source.relative_to(ROOT))
            lines.append(LINK.sub(rewrite,line))
        rewritten[source]=''.join(lines)
    if SOURCE.exists():shutil.rmtree(SOURCE)
    SOURCE.mkdir(parents=True)
    for source,destination in mapping.items():
        target=SOURCE/destination;target.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix=='.json':validate_public_text(source.read_text(),source.relative_to(ROOT))
        if source in rewritten:
            text=rewritten[source]
            if destination.parts[0] in ('assets','project'):text='---\nsearch:\n  exclude: true\n---\n\n'+text
            target.write_text(text)
        else:shutil.copy2(source,target)
    return len(rewritten)


class Page(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if attrs.get('id'):self.ids.add(attrs['id'])
        for key in ('href','src'):
            if attrs.get(key):self.links.append(attrs[key])


def check_site(site=SITE):
    pages={}
    for path in site.rglob('*.html'):
        page=Page();page.feed(path.read_text());pages[path.resolve()]=page
    errors=[]
    for path,page in pages.items():
        for target in page.links:
            parts=urlsplit(target)
            if parts.scheme or parts.netloc:continue
            destination=(site/unquote(parts.path).lstrip('/') if parts.path.startswith('/') else path.parent/unquote(parts.path)).resolve() if parts.path else path
            if destination.is_dir():destination=destination/'index.html'
            if not destination.is_file():errors.append(f'{path.relative_to(site)} → {target} (missing file)');continue
            if parts.fragment and destination in pages and unquote(parts.fragment) not in pages[destination].ids:
                errors.append(f'{path.relative_to(site)} → {target} (missing section)')
    if errors:raise ValueError('Broken manual links:\n'+'\n'.join(sorted(set(errors))))
    if not (site/'index.html').is_file() or not (site/'search.json').is_file():raise ValueError('Manual homepage/search index missing.')
    shim=site/'assets/vendor/iframe-worker/shim.js'
    if not shim.is_file():raise ValueError('Offline search shim missing.')
    index=(site/'index.html').read_text()
    if 'unpkg.com' in index:raise ValueError('Offline manual still depends on an external search shim.')
    print(f'Checked {len(pages)} rendered pages: local files, images, navigation and section links resolve.')
    return len(pages)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','prepare','check','serve'],nargs='?',default='build')
    args=parser.parse_args()
    if args.command=='check':check_site();return
    count=prepare();print(f'Staged {count} maintained Markdown pages without changing their sources.',flush=True)
    if args.command=='prepare':return
    import importlib.util
    if importlib.util.find_spec('zensical') is None:parser.error('Install the optional tools: uv pip install --python .venv/bin/python -e ".[docs]"')
    if SITE.exists():shutil.rmtree(SITE) # Rebuild only our generated site, avoiding stale pages after moves.
    if args.command=='serve':
        # Serving the built site keeps the offline and preview content identical.
        subprocess.run([sys.executable,'-m','zensical','build','--strict'],cwd=ROOT,check=True)
        check_site();subprocess.run([sys.executable,'-m','http.server','8765','--bind','127.0.0.1','--directory',str(SITE)],check=True)
    else:
        subprocess.run([sys.executable,'-m','zensical','build','--strict'],cwd=ROOT,check=True);check_site()


if __name__=='__main__':main()
