"""Preserve formatted Word documents while promoting corrected RQ4 content.

Run after build_paper_docx.sh. Unrelated paragraphs, equations, tables, styles,
sections, author metadata and image extents are retained from each destination.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
import json
import re
from lxml import etree as E

ROOT=Path(__file__).resolve().parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W='{'+NS['w']+'}'
def text(p): return ''.join(p.xpath('.//w:t/text()|.//m:t/text()',namespaces=NS))
def parts(path):
    with ZipFile(path) as z:return {n:z.read(n) for n in z.namelist()}
def settext(p,s):
    props=deepcopy(p.find(W+'pPr'))
    rpr=p.find('.//'+W+'rPr'); rpr=deepcopy(rpr) if rpr is not None else None
    for c in list(p):p.remove(c)
    if props is not None:p.append(props)
    r=E.SubElement(p,W+'r')
    if rpr is not None:r.append(rpr)
    t=E.SubElement(r,W+'t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=s
def first(ps,prefix):return next(p for p in ps if text(p).startswith(prefix))

def update(path,src):
    pkg=parts(path); doc=E.fromstring(pkg['word/document.xml']); body=doc.find('w:body',NS)
    ps=body.findall('w:p',NS); sp=src.find('w:body',NS).findall('w:p',NS)
    ieee='IEEE' in path.name
    def wording(s):
        if ieee:
            for a,b in [('Table 4','Table IV'),('Table 5','Table V'),('Figure 6','Fig. 6'),('Figure 7','Fig. 7')]:s=s.replace(a,b)
        return s
    changes=[]
    def change(p,new):
        old=text(p);new=wording(new)
        if old!=new:settext(p,new);changes.append({'before':old,'after':new})
    # Replace only the RQ4 portion of humanized abstract, findings and conclusion.
    replacement='The corrected direct membership-inference evaluation detects no resolved clean IID signal across encodings; calibrated Laplace noise reduces attribution utility but does not establish effective privacy mitigation. These attack-specific findings do not certify privacy. '
    for p in ps:
        t=text(p)
        if 'Direct membership-inference AUC on the explanation vectors is highest' in t:
            a=t.index('Direct membership-inference AUC');b=t.index('On the harder dataset',a);change(p,t[:a]+replacement+t[b:])
        elif t.startswith('Once federated and centralized'):
            a=t.index('Releasing the explanation vectors');b=t.index('A natural objection',a);change(p,t[:a]+replacement+t[b:])
        elif t.startswith('We asked whether the explanations'):
            a=t.index('Releasing the explanation vectors');b=t.index('Under severe label skew',a);change(p,t[:a]+replacement+t[b:])
        elif t.startswith('A privacy characterization.'):
            change(p,text(first(sp,'A corrected privacy evaluation.')))
        elif t.startswith('RQ4 (Privacy). Can the released explanation'):
            change(p,text(first(sp,'RQ4 (Privacy). Can released explanation')))
        elif t.startswith('RQ4 (Privacy). Can the released attribution'):
            change(p,text(first(sp,'RQ4 (Privacy). Can released attribution')))
        elif t.startswith('ESQFL [7] is'):
            a=t.index('we measure, per encoding');change(p,t[:a]+'we evaluate empirical membership discrimination and its response to calibrated release noise, without claiming certified differential privacy.')
        elif t.startswith('This work extends Chen and Yoo'):
            a=t.index('and measures how much privacy budget');change(p,t[:a]+'and evaluates empirical membership discrimination and the utility cost of calibrated release noise without deriving a required privacy budget.')
        elif t.startswith(('For anyone deploying explainable','Two implications follow for anyone')):
            change(p,text(first(sp,'Two practical qualifications')))
        elif t.startswith('The direct membership-inference AUC is a signal-presence'):
            change(p,text(first(sp,'The direct membership-inference AUC is an attack-specific')))
        elif t.startswith('The IID RQ2 and RQ4 sweeps'):
            change(p,t.replace('and every seed now enters every condition mean.','and no seed is excluded as anomalous. RQ1–RQ3 retain functional-model filtering. RQ4 IID summaries include all targets with a Track-A sensitivity check; non-IID RQ4 requires at least three Track-A targets.'))
        elif 'differential-privacy budget' in t and t.startswith('where'):
            change(p,t.replace('differential-privacy budget','calibrated-noise index'))
    # Replace the methodology block with current native Word equations.
    start=first(ps,'For RQ4, each explained input'); end=next(p for p in ps if 'Statistical Protocol' in text(p) and len(text(p))<70)
    sa=first(sp,'For RQ4, each explained input'); sb=next(p for p in sp if 'Statistical Protocol' in text(p) and len(text(p))<70)
    source_body=src.find('w:body',NS); new=list(source_body)[list(source_body).index(sa):list(source_body).index(sb)]
    old=list(body)[list(body).index(start):list(body).index(end)]
    baseprops=deepcopy(start.find(W+'pPr'))
    oldheading=next(p for p in old if '6.5b ' in text(p) or text(p).startswith('G. Differential-Privacy'))
    for el in new:
        el=deepcopy(el)
        if '6.5b Calibrated' in text(el):
            clone=deepcopy(oldheading);settext(clone,('G. ' if ieee else '6.5b ')+'Calibrated Laplace-Noise Evaluation');el=clone
        elif el.tag==W+'p' and el.find('m:oMathPara',NS) is None and el.find('m:oMath',NS) is None:
            prop=el.find(W+'pPr')
            if prop is not None:el.remove(prop)
            if baseprops is not None:el.insert(0,deepcopy(baseprops))
        body.insert(list(body).index(start),el)
    for el in old:body.remove(el)
    changes.append({'block':'RQ4 methods, equations 11–12','action':'replaced from canonical corrected DOCX'})
    # Results/captions remain in their existing positions; tables retain widths.
    ps=body.findall('w:p',NS)
    prefixes={
      'At 15 seeds under IID Wisconsin':'The corrected clean IID evaluation',
      'Two features of the direct attack':'The IID summary retains',
      'The calibrated-Laplace-noise sweep covers':'The corrected noise sweep',
      'Under non-IID conditions on Wisconsin':'For Wisconsin non-IID conditions',
    }
    for p in ps:
        t=text(p)
        for oldpref,newpref in prefixes.items():
            if t.startswith(oldpref):change(p,text(first(sp,newpref)))
        if t.endswith('RQ4: Privacy Leakage'):change(p,('D. ' if ieee else '7.4 ')+'RQ4: Corrected Release Evaluation')
        if t.startswith(('Table 4. RQ4 IID','RQ4 IID direct')):
            change(p,('' if ieee else 'Table 4. ')+'Corrected clean direct-MIA AUC, mean ± SD across 15 IID target seeds. All targets are included; Track-A sensitivity is reported in the text.')
        if t.startswith(('Table 5. RQ4 direct','RQ4 direct MIA AUC under')):
            change(p,('' if ieee else 'Table 5. ')+'Corrected calibrated Laplace sweep: direct-MIA AUC, mean ± SD across five Track-A targets per cell. Epsilon indexes noise, not a certified privacy budget.')
        if 'RQ4 membership-inference signal in the explanation vectors.' in t:
            s=text(first(sp,'Corrected direct membership-inference evaluation.'))
            change(p,('Fig. 6. ' if ieee else '')+s)
        if 'RQ4 privacy–utility frontier of explanation release.' in t:
            s=text(first(sp,'Corrected calibrated Laplace-noise evaluation on Wisconsin'))
            change(p,('Fig. 7. ' if ieee else '')+s)
    anchor=first(body.findall('w:p',NS),'For Wisconsin non-IID conditions')
    for pref in ['Stronger noise degrades explanation utility:','Attack advantage and direction-aware AUC']:
        new=deepcopy(first(sp,pref)); prop=new.find(W+'pPr')
        if prop is not None:new.remove(prop)
        if baseprops is not None:new.insert(0,deepcopy(baseprops))
        body.insert(list(body).index(anchor),new)
    dest_tables=body.findall('.//w:tbl',NS); source_tables=src.findall('.//w:tbl',NS)
    for ti in [3,4]:
        dr=dest_tables[ti].findall('w:tr',NS);sr=source_tables[ti].findall('w:tr',NS)
        assert len(dr)==len(sr)
        for drow,srow in zip(dr,sr):
            dc=drow.findall('w:tc',NS);sc=srow.findall('w:tc',NS);assert len(dc)==len(sc)
            for d,s in zip(dc,sc):
                ps_=d.findall('w:p',NS);settext(ps_[0],text(s))
                for p in ps_[1:]:d.remove(p)
    # Confidence-grading row only: RQ1–RQ3 remain byte-preserved.
    drow=dest_tables[5].findall('w:tr',NS)[-1]; srow=source_tables[5].findall('w:tr',NS)[-1]
    for d,s in zip(drow.findall('w:tc',NS),srow.findall('w:tc',NS)):
        pp=d.findall('w:p',NS);settext(pp[0],text(s))
        for p in pp[1:]:d.remove(p)
    # Replace only images identified by their neighbouring RQ4 captions.
    rels=E.fromstring(pkg['word/_rels/document.xml.rels'])
    targets={r.get('Id'):'word/'+r.get('Target') for r in rels}
    previous=[]
    for p in body.findall('w:p',NS):
        embeds=p.xpath('.//a:blip/@r:embed',namespaces=NS)
        if embeds:previous=embeds
        t=text(p)
        for prefix,stem in [('Corrected direct membership-inference evaluation.','fig06_membership'),('Corrected calibrated Laplace-noise evaluation on Wisconsin','fig07_privacy_utility')]:
            if prefix in t:
                assert len(previous)==1,(path,prefix,previous)
                pkg[targets[previous[0]]]=(ROOT/f'results/outputs/final/{stem}.png').read_bytes()
    pkg['word/document.xml']=E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for n,data in pkg.items():z.writestr(n,data)
    return changes

def main():
    src=E.fromstring(parts(ROOT/'XQFL_paper.docx')['word/document.xml'])
    files=['XQFL_paper_humanized.docx','PAPER_FORMAT/XQFL_paper_humanized.docx','PAPER_FORMAT/XQFL_IEEE_Transactions_Formatted.docx']
    log={f:update(ROOT/f,src) for f in files}
    (ROOT/'results/rq4_release_repair_v1/word_change_log.json').write_text(json.dumps(log,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
