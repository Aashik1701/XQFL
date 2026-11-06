"""Update report generator data and RQ4-only prose from current summaries."""
from pathlib import Path
import json
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.current_rq4 import mia_table, noise_table
from analysis.update_rq4_manuscript import ORDER,LABEL

def main():
    m,n=mia_table(),noise_table()
    rq4=[]
    for d,e in [('wisconsin',e) for e in ORDER]+[('pima','amplitude')]:
        r=m.query('dataset == @d and encoding == @e and distribution == "iid"').iloc[0]
        rq4.append([d.title(),LABEL[e],str(r.n),f'{r.auc_mean:.3f} ± {r.auc_std:.3f}',f'{r.n_track_a}/{r.n_total}','No'])
    non=[]
    for dist in ['qty_skew','alpha_0.1']:
        row=['Quantity skew' if dist=='qty_skew' else 'α = 0.1']
        for e in ORDER:
            r=m.query('dataset == "wisconsin" and encoding == @e and distribution == @dist').iloc[0]
            row.append((f'{r.auc_mean:.3f}' if r.status=='ok' else '—')+f' ({r.n}/5)')
        non.append(row)
    noise=[]
    for eps in [.1,.5,1,5]:
        row=[f'{eps:g}']
        for e in ORDER:
            r=n.query('encoding == @e and epsilon == @eps').iloc[0];row.append(f'{r.auc_mean:.3f} ± {r.auc_std:.3f}')
        noise.append(row)
    (ROOT/'report_build/assets/rq4_current.json').write_text(json.dumps(dict(RQ4=rq4,RQ4NONIID=non,RQ4DP=noise),indent=2)+'\n')
    p=ROOT/'report_build/ch6.js';s=p.read_text()
    a=s.index('const RQ4 =');b=s.index('const COMPARE',a)
    s=s[:a]+'const { RQ4, RQ4NONIID, RQ4DP } = require("./assets/rq4_current.json");\n\n'+s[b:]
    a=s.index('    L.sec("6.4",');b=s.index('    L.sec("6.6",',a)
    s=s[:a]+'''    L.sec("6.4", "RQ4: Corrected Membership-Inference Evaluation"),
    L.p("The corrected release evaluation does not reproduce the previous leakage ranking. No clean IID mean is resolved above chance under the study's two-standard-error convention. Table 6.5 retains all fifteen IID targets per group. Wisconsin Angle and Pima Amplitude each have fourteen Track-A targets; Track-A-only sensitivity does not alter the conclusion. Absence of detected discrimination by this attack does not establish privacy."),
    L.tabCaption("Table 6.5  Corrected clean direct-MIA AUC, mean ± SD across fifteen target seeds. Chance is 0.5; the final column tests the mean against chance, not against Angle."),
    L.table(["Dataset", "Encoding", "n", "AUC (mean ± SD)", "Track A", "Above chance at 2 SE?"], RQ4,
      [15, 18, 8, 25, 18, 16], { size: 20, align: [null, null, C, C, C, C] }),
    L.figure(A("figure5_mia_auc_comparison.png"), 5.6, [2070, 1230]),
    L.figCaption("Figure 6.3  Corrected direct-MIA AUC. Points and error bars show mean ± SD across fifteen IID targets; the dashed line is chance. No encoding has a resolved clean IID signal. This is an attack-specific signal probe, not a realistic attacker-success rate or a privacy guarantee."),
    L.p("Both member and non-member explanations now use the same twenty-record training-derived background, disjoint from the fifty queried members. Standardization and logistic regression are fitted only within training folds. The primary statistic is the mean of five held-out fold AUCs. Four inner training folds select score orientation and a TPR-minus-FPR threshold for a separate direction-aware evaluation. Scores, labels and fold assignments are retained. Historical shadow-model scores are not mixed with the corrected protocol."),
    L.p("Non-IID summaries retain Track-A targets only and require at least three. Angle under severe label skew is insufficient at one of five targets. The corrected results do not establish the previous quantity-skew leakage leadership or a stable encoding ranking. Pima non-IID attacks remain untested."),
    L.tabCaption("Table 6.6  Corrected Wisconsin non-IID direct-MIA AUC means. Parentheses give functional targets out of five; a dash denotes insufficient coverage, not zero."),
    L.table(["Regime", "Angle", "Amplitude", "Re-uploading", "IQP"], RQ4NONIID,
      [24, 19, 19, 19, 19], { size: 20, align: [null, C, C, C, C] }),

    L.sec("6.5", "RQ4: Calibrated Release-Noise Evaluation"),
    L.p("The corrected sweep evaluates four noise levels for each of five Wisconsin IID targets per encoding. Each full set of one hundred per-sample explanations receives an independently sampled noise matrix, without cross-group reuse. The target stays fixed across release settings. Utility is measured on the same clean/noisy records scored by the attack, rather than separately simulated averaged explanations."),
    L.tabCaption("Table 6.7  Corrected direct-MIA AUC under calibrated Laplace noise, mean ± SD across five Track-A targets per cell. Epsilon is a noise-level index, not a certified privacy budget."),
    L.table(["Index ε", "Angle", "Amplitude", "Re-uploading", "IQP"], RQ4DP,
      [14, 21, 22, 22, 21], { size: 20, align: [C, C, C, C, C] }),
    L.figure(A("figure6_dp_auc_vs_epsilon.png"), 5.6, [2070, 1230]),
    L.figCaption("Figure 6.4  Corrected calibrated Laplace-noise response. Points and error bars give raw direct-MIA AUC mean ± SD across five targets, with categorical tested indices and a chance line at 0.5. No monotonic mitigation is established. The empirical scale is not a certified sensitivity bound; no formal privacy guarantee is claimed."),
    L.p("The sweep does not reproduce the earlier monotonic suppression claim. The isolated IQP cell at epsilon = 1 has AUC 0.561 ± 0.051, approximately 2.65 standard errors above chance in an unadjusted comparison; it is exploratory, not a stable ranking. Matched cosine utility spans 0.013–0.063 at epsilon = 0.1 and 0.701–0.829 at epsilon = 5. Stronger noise degrades the explanation, but clean baselines already lack a resolved signal, so successful mitigation is not established."),
    L.p("Post-hoc max(AUC, 1-AUC) is an orientation diagnostic, not a validated attack score. Attack advantage is held-out TPR minus FPR at a training-selected threshold; negative estimates are retained. With fifty non-members, pooled empirical FPR resolution is 0.02. Low-FPR ROC values are descriptive and do not validate one-percent-FPR guarantees."),
    L.p("Answer to RQ4: the evaluated attack does not establish clean IID membership discrimination or an encoding-dependent leakage ranking. The matched release experiment measures utility degradation but establishes neither effective privacy mitigation nor formal differential privacy."),

'''+s[b:]
    a=s.index('  ["RQ4 Privacy",');b=s.index('\n];',a)
    s=s[:a]+'  ["RQ4 Privacy", "Limited: clean IID n=15; no detected discrimination; Pima check", "Limited: clean IID n=15; no detected discrimination", "Limited: clean IID n=15, Track A 14/15; no detected discrimination", "Limited: clean IID n=15; isolated noisy cell exploratory"],'+s[b:]
    a=s.index('    L.bullet([{ text: "Noise calibration must');b=s.index('    L.bullet([{ text: "Unequal data',a)
    s=s[:a]+'''    L.bullet([{ text: "No effective privacy budget is established. ", bold: true },
      { text: "The corrected attack has no resolved clean IID signal. Noise decreases attribution utility, but the evaluation does not justify a deployment protection rule." }]),
    L.bullet([{ text: "A null attack is not a privacy certificate. ", bold: true },
      { text: "Stronger attacks, independent reference data and larger evaluation samples may detect risks not resolved by this logistic-regression probe." }]),
'''+s[b:]
    s=s.replace('the noise budget actually required to suppress an explanation-based attack is measured per encoding','membership discrimination and matched release-noise utility are evaluated without a certified privacy guarantee')
    s=s.replace('Privatises the explanation release itself, with the mechanism calibrated from measured attribution magnitude and the trade-off traced across four budgets','Perturbs explanation releases using an empirical attribution scale and evaluates their matched utility; no formal privacy is established')
    p.write_text(s)
    # Replace concatenated JS string expressions while preserving surrounding structure.
    pattern=re.compile(r'"(?:[^"\\]|\\.)*"(?:\s*\+\s*"(?:[^"\\]|\\.)*")+')
    for name in ['front','ch1','ch3','ch5','ch7']:
        p=ROOT/f'report_build/{name}.js';s=p.read_text()
        def transform(match):
            value=''.join(json.loads(v) for v in re.findall(r'"(?:[^"\\]|\\.)*"',match.group()))
            new=value
            if 'Releasing the explanation vectors is not free:' in value:
                new=value[:value.index('Releasing the explanation vectors')]+'The corrected direct attack detects no resolved clean IID membership signal. Calibrated noise reduces attribution utility, but neither effective mitigation nor a certified privacy guarantee is established.'
            elif 'four-budget calibrated-noise sweep shows' in value:
                new='A corrected fold-isolated direct-MIA protocol evaluates fifteen IID targets per group and a four-level calibrated release-noise sweep. It detects no resolved clean IID discrimination; matched noise utility is measured without claiming effective mitigation.'
            elif value.startswith('Releasing the explanation vectors raises direct membership'):
                new='The corrected clean IID attack does not establish membership discrimination or an encoding ranking. Noise reduces matched explanation utility but does not establish privacy mitigation.'
            elif value.startswith('Two of these deserve emphasis'):
                new='Two limitations matter for deployment: a null result from one attack family is not a privacy guarantee, and collapsed models require a functional-model gate before standard attribution comparisons.'
            elif 'its standardisation is fitted on the pooled set' in value:
                new='The corrected pipeline fits standardization within training folds and uses one common training-derived background. The attack still uses labelled explanations from the same target, and only one attack family with one hundred queried records is evaluated. A null result does not establish privacy.'
            elif value.startswith('The alternative of a conservative theoretical bound'):
                new='An earlier release evaluation reused noise across groups, standardized before cross-validation and used different explanation backgrounds. All 115 RQ4 targets and 195 releases were regenerated under the corrected protocol. Below-chance AUC is not evidence of privacy, and the earlier mitigation interpretation is withdrawn. RQ1–RQ3 are unchanged.'
            elif value.startswith('Table 5.3 summarises'):
                new='Table 5.3 links each research question to its metric. RQ1–RQ3 use functional runs. Corrected RQ4 IID includes all targets with a Track-A-only sensitivity check; non-IID RQ4 requires at least three functional targets.'
            elif value.startswith('Each explained input carries a membership label'):
                new='Each input has membership label one for a training record and zero for a test record. Fifty of each are explained against a common twenty-record training background disjoint from queried members. Standardization and logistic regression are fitted within each of five stratified outer training folds. The primary AUC is the arithmetic mean of held-out fold AUCs:'
            return json.dumps(new,ensure_ascii=False) if new!=value else match.group()
        s=pattern.sub(transform,s)
        if name=='ch3':
            a=s.index('    L.eq([RM("AUC")');b=s.index('    L.p("A value of one half',a)
            s=s[:a]+'    L.eq([RM("AUC"), SUB("MIA"), RM(" = (1/5) Σ"), SUB("h=1..5"), RM(" AUC(y"), SUB("Ih"), RM(", g"), SUB("-h"), RM("(Φ"), SUB("Ih"), RM("))")], "3.13"),\n'+s[b:]
        if name=='ch5':
            a=s.index('const DELTAF =');b=s.index('\n];',a)+3
            s=s[:a]+'const DELTAF = [["Angle", "0.229"], ["Amplitude", "0.262"], ["Re-uploading", "0.344"], ["IQP", "0.287"]];'+s[b:]
            a=s.index('    L.tabCaption("Table 5.4');b=s.index('    L.p(',a)
            s=s[:a]+'    L.tabCaption("Table 5.4  Empirical attribution scale per encoding; not a certified sensitivity bound."),\n    L.table(["Encoding", "Empirical scale"], DELTAF, [50, 50], { size: 20, align: [null, C] }),\n'+s[b:]
            s=s.replace('0.5 is chance; higher means more leakage','0.5 is chance; below-chance ranking is not privacy')
            s=s.replace('Traces the privacy-utility curve of explanation release','Measures attack response and matched release utility')
        p.write_text(s)
    for pattern in ['table4*','table5*','figure5_mia*','figure6_dp*']:
        for src in (ROOT/'results/outputs').glob(pattern):shutil.copy2(src,ROOT/'report_build/assets'/src.name)

if __name__=='__main__':main()
