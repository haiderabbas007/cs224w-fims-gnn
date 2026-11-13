#!/usr/bin/env python3
"""Install a verified immutable code package; compatible with login-node Python."""
import argparse,hashlib,json,os,re,shutil,tarfile,tempfile
from pathlib import Path,PurePosixPath

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def safe(name):
    p=PurePosixPath(name)
    return not p.is_absolute() and '..' not in p.parts and '\\' not in name and str(p)==name and name not in ('','.','..')

def unpack(archive,base,expected,manifest_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected) or not re.fullmatch('[0-9a-f]{64}',manifest_sha):
        raise ValueError('Invalid expected digest')
    if sha(archive)!=expected:raise ValueError('Transfer archive checksum mismatch')
    base=Path(base).resolve();base.mkdir(parents=True,exist_ok=True)
    root=base/('study_'+manifest_sha[:12])
    with tarfile.open(archive,'r:gz') as z:
        members=z.getmembers();files=[m for m in members if m.isfile()];names=[m.name for m in files]
        if len({m.name for m in members})!=len(members) or any(not safe(m.name) for m in members):
            raise ValueError('Duplicate or unsafe archive path')
        if any(not(m.isfile() or m.isdir()) for m in members):raise ValueError('Archive links and devices are forbidden')
        if sum(m.size for m in files)>256*1024*1024:raise ValueError('Code package unexpectedly large')
        raw=z.extractfile('PACKAGE_MANIFEST.json').read()
        if hashlib.sha256(raw).hexdigest()!=manifest_sha:raise ValueError('Manifest checksum mismatch')
        manifest=json.loads(raw);inventory=manifest['files']
        if manifest.get('schema')!='fims_operating_point_review_package_v1' or not isinstance(inventory,dict) or not inventory:
            raise ValueError('Wrong operating-point package schema')
        if set(names)!=set(inventory)|{'PACKAGE_MANIFEST.json'}:raise ValueError('Archive inventory mismatch')
        if any(not safe(n) for n in inventory):raise ValueError('Unsafe manifest path')
        if root.exists():
            if root.is_symlink() or not root.is_dir():raise ValueError('Unsafe existing study path')
            for n,d in dict(inventory,**{'PACKAGE_MANIFEST.json':manifest_sha}).items():
                p=root/n
                if p.is_symlink() or not p.is_file() or os.path.commonpath([str(root),str(p.resolve())])!=str(root) or sha(p)!=d:
                    raise ValueError('Existing package changed: '+n)
            return root
        stage=Path(tempfile.mkdtemp(prefix='.install_partial_',dir=str(base)))
        try:
            for m in files:
                p=stage/m.name;p.parent.mkdir(parents=True,exist_ok=True)
                with z.extractfile(m) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,1048576)
                digest=manifest_sha if m.name=='PACKAGE_MANIFEST.json' else inventory[m.name]
                if sha(p)!=digest:raise ValueError('Extracted source checksum mismatch: '+m.name)
            os.rename(str(stage),str(root))
        finally:
            if stage.exists():shutil.rmtree(str(stage))
    return root

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--base',required=True)
    p.add_argument('--expected-sha',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args()
    print(unpack(a.archive,a.base,a.expected_sha,a.manifest_sha))

if __name__=='__main__':main()
