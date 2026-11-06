"""Apply only generator changes to the manually formatted report via a three-way merge."""
from pathlib import Path
import tempfile,shutil,subprocess,os,json,difflib,re
from copy import deepcopy
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from update_rq4_word import parts,text,NS,W,settext
ROOT=Path(__file__).resolve().parents[1]
NODE='/Users/mohammedaashik/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
MODULES='/Users/mohammedaashik/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'

def key(el):
    t=text(el)
    if any('PAGEREF' in v for v in el.xpath('.//w:instrText/text()',namespaces=NS)):
        t=re.sub(r'\d+$','',t)
    return el.tag.split('}')[-1]+':'+re.sub(r'\s+','',t)
def main():
    env={**os.environ,'NODE_PATH':MODULES}
    tmp=Path(tempfile.mkdtemp(prefix='xqfl-report-before-',dir='/private/tmp'))
    src=ROOT/'report_build'
    shutil.copytree(src/'assets',tmp/'assets')
    for p in (ROOT/'results/pre_rq4_release_promotion_20260928/results/outputs').glob('table[45]*.csv'):
        shutil.copy2(p,tmp/'assets/tables'/p.name)
    for p in (ROOT/'results/pre_rq4_release_promotion_20260928/report_build').glob('*.js'):shutil.copy2(p,tmp/p.name)
    # Tables in assets/tables are still the pre-promotion copies at this point.
    subprocess.run([NODE,str(tmp/'build.js')],check=True,env=env,capture_output=True)
    old=E.fromstring(parts(tmp/'XQFL_Final_Project_Report.docx')['word/document.xml'])
    # Refresh the actual appendix input location, not the redundant assets root.
    for p in (ROOT/'results/outputs').glob('table[45]*.csv'):shutil.copy2(p,src/'assets/tables'/p.name)
    subprocess.run([NODE,str(src/'build.js')],check=True,env=env,capture_output=True)
    newpkg=parts(src/'XQFL_Final_Project_Report.docx');new=E.fromstring(newpkg['word/document.xml'])
    dest=ROOT/'XQFL_Final_Project_Report.docx';pkg=parts(dest);doc=E.fromstring(pkg['word/document.xml'])
    ob,nb,db=[r.find('w:body',NS) for r in [old,new,doc]]
    o,n=list(ob),list(nb);ok,nk=[key(e) for e in o],[key(e) for e in n]
    # Add only generated image relationships used by inserted/replaced content.
    R='{http://schemas.openxmlformats.org/package/2006/relationships}'
    rels=E.fromstring(pkg['word/_rels/document.xml.rels']); nr=E.fromstring(newpkg['word/_rels/document.xml.rels'])
    imgmap={}
    for r in nr:
        if r.get('Type','').endswith('/image'):
            rid='rIdRQ4'+str(len(imgmap)+1);target='media/rq4_'+Path(r.get('Target')).name
            pkg['word/'+target]=newpkg['word/'+r.get('Target')]
            clone=deepcopy(r);clone.set('Id',rid);clone.set('Target',target);rels.append(clone);imgmap[r.get('Id')]=rid
    def clone(el):
        c=deepcopy(el)
        for b in c.findall('.//a:blip',NS):b.set('{'+NS['r']+'}embed',imgmap[b.get('{'+NS['r']+'}embed')])
        return c
    log=[]
    for tag,a,b,c,d in difflib.SequenceMatcher(a=ok,b=nk,autojunk=False).get_opcodes():
        if tag=='equal':continue
        # Find the old generated nodes independently; retain unrelated manual content.
        existing=[]
        cursor=0
        for node in o[a:b]:
            if not text(node).strip():continue
            candidates=[el for el in list(db)[cursor:] if key(el)==key(node)]
            if len(candidates)!=1:
                raise RuntimeError(f'Unmatched/ambiguous manual report content: {key(node)[:160]} ({len(candidates)})')
            el=candidates[0];existing.append(el);cursor=list(db).index(el)+1
        if not existing:
            # Pure insertion: anchor at the next stable text-bearing source element.
            anchor=next((el for oldnode in o[b:] if text(oldnode).strip() for el in db if key(el)==key(oldnode)),None)
            if anchor is None:raise RuntimeError('Cannot locate report insertion anchor')
            position=list(db).index(anchor)
        else:position=list(db).index(existing[0])
        for node in n[c:d]:db.insert(position,clone(node));position+=1
        for el in existing:db.remove(el)
        log.append(dict(kind=tag,before=[text(e) for e in o[a:b]],after=[text(e) for e in n[c:d]]))
    # Refresh existing figure media independently (image-only nodes compare equal).
    origrels={r.get('Id'):r.get('Target') for r in rels}
    previous=[]
    for p in db.findall('w:p',NS):
        embeds=p.xpath('.//a:blip/@r:embed',namespaces=NS)
        if embeds:previous=embeds
        t=text(p)
        for prefix,stem in [('Figure 6.3  Corrected','figure5_mia_auc_comparison'),('Figure 6.4  Corrected','figure6_dp_auc_vs_epsilon')]:
            if t.startswith(prefix):
                for rid in previous:pkg['word/'+origrels[rid]]=(ROOT/f'results/outputs/{stem}.png').read_bytes()
    # Bookmark/numeric IDs must remain unique after generated-block insertion.
    for tag in ['bookmarkStart','bookmarkEnd']:
        # Existing named references retained; report renderer refreshes page fields.
        pass
    docprs=doc.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr')
    for i,el in enumerate(docprs,1):el.set('id',str(i))
    pkg['word/document.xml']=E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True)
    pkg['word/_rels/document.xml.rels']=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(dest,'w',ZIP_DEFLATED) as z:
        for name,data in pkg.items():z.writestr(name,data)
    (ROOT/'results/rq4_release_repair_v1/report_word_change_log.json').write_text(json.dumps(log,indent=2,ensure_ascii=False)+'\n')
    print('Report merged; personal front matter retained:',len(log),'changed blocks')
if __name__=='__main__':main()
